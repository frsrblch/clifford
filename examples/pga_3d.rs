fn main() {}

clifford::algebra! {
    bases {
        x ^ 2 == 1,
        y ^ 2 == 1,
        z ^ 2 == 1,
        w ^ 2 == 0,
    }
}

#[test]
fn unit_vector_dual() {
    let u = Vector::new(1., 2., 3., 4.).unit();
    assert_eq!(
        Trivector::new(4f64, -3., 2., -1.) / f64::sqrt(14.0),
        u.left_comp()
    );
}
