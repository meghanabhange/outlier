use ciphercore_base::custom_ops::CustomOperation;
use ciphercore_base::graphs::Graph;
use ciphercore_base::graphs::Node;
use ciphercore_base::graphs::SliceElement;
use ciphercore_base::ops::newton_inversion::NewtonInversion;

pub struct LinearRegression {
    pub coefficients: Node,
    pub slope: Node,
    pub intercept: Node,
}

impl LinearRegression {
    pub fn new(graph: &Graph, x: Node, y: Node) -> Self {
        let axes = vec![1, 0];

        let x_t = x.permute_axes(axes).unwrap();
        let x_t_x = x_t.matmul(x).unwrap();

        let x_t_x_inv = graph
            .custom_op(
                CustomOperation::new(NewtonInversion {
                    iterations: 100,
                    denominator_cap_2k: 4,
                }),
                vec![x_t_x],
            )
            .unwrap();
        let x_t_x_inv_x_t = x_t_x_inv.matmul(x_t).unwrap();
        let coefficients = x_t_x_inv_x_t.matmul(y).unwrap();
        let slope = coefficients
            .get_slice(vec![SliceElement::Ellipsis, SliceElement::SingleIndex(0)])
            .unwrap();
        let intercept = coefficients
            .get_slice(vec![SliceElement::Ellipsis, SliceElement::SingleIndex(1)])
            .unwrap();

        Self {
            coefficients,
            slope,
            intercept,
        }
    }
}
