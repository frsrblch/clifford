fn main() {}

clifford::algebra! {
    bases {
        x ^ 2 == 1,
        y ^ 2 == 1,
        z ^ 2 == 1,
    }
}

#[test]
fn sum_test() {
    let u = Vector::new(1., 2., 3.).unit();
    let three: Vector = [u, u, u].into_iter().sum();
    assert_eq!(three, u * 3.0);
}

#[test]
fn vector_unit() {
    let v = Vector::new(3., 4., 0.);
    let u = Vector::new(0.6, 0.8, 0.0).unit();
    assert_eq!(u, v.unit());
}

#[test]
fn vector_inv() {
    let v = Vector::new(3., 4., 0.);
    let u = Vector::new(0.12, 0.16, 0.0);
    assert_eq!(u, v.inv());
}

#[test]
fn motor_dot_product() {
    let a = Motor::new(1.0, 2.0, 3.0, 4.0);
    let b = Motor::new(3.0, 5.0, 7.0, 11.0);
    assert_eq!(a.dot(b), -44.0 - 21.0 - 10.0 + 3.0);
}
