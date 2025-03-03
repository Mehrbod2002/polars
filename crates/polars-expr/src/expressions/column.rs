use std::borrow::Cow;

use polars_core::prelude::*;
use polars_plan::constants::CSE_REPLACED;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct ColumnExpr {
    name: PlSmallStr,
    expr: Expr,
    schema: SchemaRef,
}

pub struct MultiColumnExpr {
    names: Vec<PlSmallStr>,
    expr: Expr,
    schema: SchemaRef,
}

impl MultiColumnExpr {
    pub fn new(names: Vec<PlSmallStr>, expr: Expr, schema: SchemaRef) -> Self {
        Self {
            names,
            expr,
            schema,
        }
    }
}

impl MultiColumnExpr {
    fn check_external_context(
        &self,
        out: PolarsResult<Vec<Column>>,
        state: &ExecutionState,
    ) -> PolarsResult<Vec<Column>> {
        match out {
            Ok(cols) => Ok(cols),
            Err(e) => {
                if state.ext_contexts.is_empty() {
                    Err(e)
                } else {
                    if let Some(df) = state.ext_contexts.as_ref().iter().next() {
                        let mut collected = Vec::new();
                        for name in &self.names {
                            if let Ok(col) = df.column(name) {
                                collected.push(col.clone());
                            } else {
                                return Err(e);
                            }
                        }
                        return Ok(collected);
                    }
                    Err(e)
                }
            },
        }
    }

    fn process_by_idx(
        &self,
        _out: &[Column],
        _state: &ExecutionState,
        _schema: &Schema,
        df: &DataFrame,
        check_state_schema: bool,
    ) -> PolarsResult<Vec<Column>> {
        let mut collected = Vec::new();
        for name in &self.names {
            if let Ok(col) = df.column(name) {
                collected.push(col.clone());
            } else if check_state_schema {
                if let Some(schema) = _state.get_schema() {
                    return self.process_from_state_schema(df, _state, &schema);
                }
            } else {
                return Err(PolarsError::ColumnNotFound(name.to_string().into()));
            }
        }
        Ok(collected)
    }

    fn process_by_linear_search(
        &self,
        df: &DataFrame,
        _state: &ExecutionState,
        _panic_during_test: bool,
    ) -> PolarsResult<Vec<Column>> {
        self.names
            .iter()
            .map(|name| df.column(name).cloned())
            .collect()
    }

    fn process_from_state_schema(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        schema: &Schema,
    ) -> PolarsResult<Vec<Column>> {
        let mut collected = Vec::new();
        for name in &self.names {
            match schema.get_full(name) {
                None => return self.process_by_linear_search(df, state, true),
                Some((idx, _, _)) => match df.get_columns().get(idx) {
                    Some(out) => collected.push(out.clone()),
                    None => return self.process_by_linear_search(df, state, true),
                },
            }
        }
        Ok(collected)
    }

    fn process_cse(&self, df: &DataFrame, schema: &Schema) -> PolarsResult<Vec<Column>> {
        // The CSE columns are added on the rhs.
        let offset = schema.len();
        let columns = &df.get_columns()[offset..];
        // Linear search will be relatively cheap as we only search the CSE columns.
        let mut collected = Vec::new();
        for name in &self.names {
            if let Some(col) = columns.iter().find(|s| s.name() == name) {
                collected.push(col.clone());
            } else {
                return Err(PolarsError::ColumnNotFound(name.to_string().into()));
            }
        }
        Ok(collected)
    }
}

impl ColumnExpr {
    pub fn new(name: PlSmallStr, expr: Expr, schema: SchemaRef) -> Self {
        Self { name, expr, schema }
    }
}

impl ColumnExpr {
    fn check_external_context(
        &self,
        out: PolarsResult<Column>,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        match out {
            Ok(col) => Ok(col),
            Err(e) => {
                if state.ext_contexts.is_empty() {
                    Err(e)
                } else {
                    if let Some(df) = state.ext_contexts.as_ref().iter().next() {
                        let out = df.column(&self.name);
                        if out.is_ok() {
                            return out.cloned();
                        }
                    }
                    Err(e)
                }
            },
        }
    }

    fn process_by_idx(
        &self,
        out: &Column,
        _state: &ExecutionState,
        _schema: &Schema,
        df: &DataFrame,
        check_state_schema: bool,
    ) -> PolarsResult<Column> {
        if out.name() != &*self.name {
            if check_state_schema {
                if let Some(schema) = _state.get_schema() {
                    return self.process_from_state_schema(df, _state, &schema);
                }
            }

            df.column(&self.name).cloned()
        } else {
            Ok(out.clone())
        }
    }
    fn process_by_linear_search(
        &self,
        df: &DataFrame,
        _state: &ExecutionState,
        _panic_during_test: bool,
    ) -> PolarsResult<Column> {
        df.column(&self.name).cloned()
    }

    fn process_from_state_schema(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        schema: &Schema,
    ) -> PolarsResult<Column> {
        match schema.get_full(&self.name) {
            None => self.process_by_linear_search(df, state, true),
            Some((idx, _, _)) => match df.get_columns().get(idx) {
                Some(out) => self.process_by_idx(out, state, schema, df, false),
                None => self.process_by_linear_search(df, state, true),
            },
        }
    }

    fn process_cse(&self, df: &DataFrame, schema: &Schema) -> PolarsResult<Column> {
        // The CSE columns are added on the rhs.
        let offset = schema.len();
        let columns = &df.get_columns()[offset..];
        // Linear search will be relatively cheap as we only search the CSE columns.
        Ok(columns
            .iter()
            .find(|s| s.name() == &self.name)
            .unwrap()
            .clone())
    }
}

impl PartitionedAggregation for MultiColumnExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let columns = self.evaluate(df, state)?;
        Ok(columns)
    }

    fn finalize(
        &self,
        partitioned: Column,
        _groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        Ok(partitioned)
    }
}

impl PhysicalExpr for MultiColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut columns = Vec::new();

        for name in &self.names {
            let col = match self.schema.get_full(name) {
                Some((idx, _, _)) => {
                    // Check if the schema was correct, otherwise do O(n) search
                    match df.get_columns().get(idx) {
                        Some(out) => {
                            self.process_by_idx(&[out.clone()], state, &self.schema, df, true)
                        },
                        None => {
                            // Handle partitioned group_by special case
                            if let Some(schema) = state.get_schema() {
                                self.process_from_state_schema(df, state, &schema)
                            } else {
                                self.process_by_linear_search(df, state, true)
                            }
                        },
                    }
                },
                None => {
                    // Handle CSE_REPLACED case
                    if name.starts_with(CSE_REPLACED) {
                        return self
                            .process_cse(df, &self.schema)
                            .map(|mut cols| cols.remove(0));
                    }
                    self.process_by_linear_search(df, state, true)
                },
            }?;
            columns.push(col);
        }

        let columns =
            self.check_external_context(Ok(columns.into_iter().flatten().collect()), state)?;
        // Assuming we need to return the first column for compatibility
        columns
            .into_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("No columns found".into()))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let cols = self.evaluate(df, state)?;
        Ok(AggregationContext::new(cols, Cow::Borrowed(groups), false))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        let name = self.names.first().ok_or_else(|| {
            polars_err!(
                ColumnNotFound: "could not find any names in MultiColumnExpr"
            )
        })?;
        input_schema.get_field(name).ok_or_else(|| {
            polars_err!(
                ColumnNotFound: "could not find {:?} in schema: {:?}", name, &input_schema
            )
        })
    }

    fn is_scalar(&self) -> bool {
        false
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let out = match self.schema.get_full(&self.name) {
            Some((idx, _, _)) => {
                // check if the schema was correct
                // if not do O(n) search
                match df.get_columns().get(idx) {
                    Some(out) => self.process_by_idx(out, state, &self.schema, df, true),
                    None => {
                        // partitioned group_by special case
                        if let Some(schema) = state.get_schema() {
                            self.process_from_state_schema(df, state, &schema)
                        } else {
                            self.process_by_linear_search(df, state, true)
                        }
                    },
                }
            },
            // in the future we will throw an error here
            // now we do a linear search first as the lazy reported schema may still be incorrect
            // in debug builds we panic so that it can be fixed when occurring
            None => {
                if self.name.starts_with(CSE_REPLACED) {
                    return self.process_cse(df, &self.schema);
                }
                self.process_by_linear_search(df, state, true)
            },
        };
        self.check_external_context(out, state)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let c = self.evaluate(df, state)?;
        Ok(AggregationContext::new(c, Cow::Borrowed(groups), false))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        input_schema.get_field(&self.name).ok_or_else(|| {
            polars_err!(
                ColumnNotFound: "could not find {:?} in schema: {:?}", self.name, &input_schema
            )
        })
    }
    fn is_scalar(&self) -> bool {
        false
    }
}

impl PartitionedAggregation for ColumnExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        self.evaluate(df, state)
    }

    fn finalize(
        &self,
        partitioned: Column,
        _groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        Ok(partitioned)
    }
}
