use ciphercore_base::data_types::{array_type, Type, INT64};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::create_context;

use crate::linear::LinearRegression;

mod linear;

fn main() -> Result<()> {
    let context = create_context()?;
    let graph = context.create_graph()?;

    let t: Type = array_type(vec![3, 3], INT64);
    let y = array_type(vec![3, 1], INT64);
    let x = graph.input(t.clone()).unwrap();
    let y = graph.input(y.clone()).unwrap();

    let linear_regression = LinearRegression::new(&graph, x, y);
    let coefficients = linear_regression.coefficients;

    graph.set_output_node(coefficients)?;

    graph.finalize()?;
    context.set_main_graph(graph)?;
    context.finalize()?;
    println!("{}", serde_json::to_string(&context)?);
    Ok(())
}
