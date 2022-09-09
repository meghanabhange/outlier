use ciphercore_base::graphs::Node;
use ciphercore_base::graphs::Graph;
use ciphercore_base::ops::newton_inversion::NewtonInversion;
use ciphercore_base::custom_ops::CustomOperation;


pub fn linear_regression(graph: &Graph, x: Node, y: Node) -> Node {
    let axes = vec![1, 0];
    let x_t = x.permute_axes(axes).unwrap();
    let x_t_x = x_t.matmul(x).unwrap();

    let x_t_x_inv = graph.custom_op(CustomOperation::new(NewtonInversion { iterations: 10, denominator_cap_2k: 4 }), vec![x_t_x]).unwrap();
    let x_t_x_inv_x_t = x_t_x_inv.matmul(x_t).unwrap();
    let coefficients = x_t_x_inv_x_t.matmul(y).unwrap();
    coefficients
}