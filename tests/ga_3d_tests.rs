use clifford::ga_3d::*;

#[test]
fn g3_contains_dual() {
    let vector = Vector::new(1., 2., 3.);
    let expected = Bivector::new(3., -2., 1.);
    assert_eq!(expected, vector.dual());
}
