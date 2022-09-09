mod linear;

use ciphercore_base::graphs::create_context;
use ciphercore_base::data_types::{array_type, INT64, Type};
use ciphercore_base::errors::Result;

fn main() -> Result<()> {
        let context = create_context()?;
        let graph = context.create_graph()?;


        let t: Type = array_type(vec![3, 3], INT64);
        let y = array_type(vec![3, 1], INT64);
        let x = graph.input(t.clone()).unwrap();
        let y = graph.input(y.clone()).unwrap();


        let coefficients = linear::linear_regression(&graph, x, y);

        graph.set_output_node(coefficients)?;

        graph.finalize()?;
        context.set_main_graph(graph)?;
        context.finalize()?;
        println!("{}", serde_json::to_string(&context)?);
        Ok(())
    }
