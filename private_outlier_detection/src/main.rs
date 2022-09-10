use ciphercore_base::data_types::{array_type, Type, INT64};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::create_context;
use ciphercore_base::graphs::SliceElement;

use crate::linear::LinearRegression;

mod linear;

fn main() -> Result<()> {
    let context = create_context()?;
    let graph = context.create_graph()?;

    let t: Type = array_type(vec![76, 3], INT64);
    let data = graph.input(t.clone()).unwrap();
    let slice_x = vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, Some(2), Some(1)),
    ];
    let slice_y = vec![SliceElement::Ellipsis, SliceElement::SingleIndex(2)];

    let x = data.get_slice(slice_x).unwrap();
    let y = data.get_slice(slice_y).unwrap();

    let linear_regression = LinearRegression::new(&graph, x, y);
    let coefficients = linear_regression.coefficients;

    graph.set_output_node(coefficients)?;

    graph.finalize()?;
    context.set_main_graph(graph)?;
    context.finalize()?;
    println!("{}", serde_json::to_string(&context)?);
    Ok(())
}
