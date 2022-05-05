proc_macros::clifford!(3, 1, 0);

/// Point at the origin
pub const N: Vector = Vector::new(0., 0., 0.5, 0.5);

/// Point through infinity
pub const N_BAR: Vector = Vector::new(0., 0., -1., 1.);

pub const fn point(x: f64, y: f64) -> Vector {
    let x2 = x * x + y * y;
    Vector::new(x, y, 0.5 - 0.5 * x2, 0.5 + 0.5 * x2)
}

#[test]
fn vector_test() {
    let x = Vector::new(2., 3., 0., 0.);
    let x2 = 2. * 2. + 3. * 3.;
    let expected = x + 0.5 * x2 * N_BAR + N;

    let actual = point(2., 3.);

    assert_eq!(expected, actual);
}

trait IsFlat {
    fn is_flat(self) -> bool;
}

impl<T, U> IsFlat for T
where
    T: crate::Wedge<Vector, Output = U>,
    U: Default + PartialEq,
{
    fn is_flat(self) -> bool {
        self.wedge(N_BAR) == U::default()
    }
}

#[test]
fn flat_test() {
    let o = point(0., 0.);
    let a = point(1., 0.);
    let b = point(0., 1.);

    let points = o.wedge(a);
    let line = points.wedge(N_BAR);
    let circle = points.wedge(b);

    assert!(!o.is_flat());
    assert!(!points.is_flat());
    assert!(!circle.is_flat());
    assert!(line.is_flat());
}
