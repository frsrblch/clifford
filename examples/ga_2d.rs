fn main() {}

clifford::algebra! {
    bases {
        x ^ 2 == 1,
        y ^ 2 == 1,
    }
}

#[test]
fn motor_dot_product() {
    let a = Motor::new(1.0, 2.0);
    let b = Motor::new(3.0, 5.0);
    assert_eq!(a.dot(b), -7.0);
}
