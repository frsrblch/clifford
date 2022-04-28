#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert
)]

// TODO add efficient implementation for null cone geometries
// TODO consider grade enum model with dynamic multivectors (e.g., BTreeMap<Grade, f64>)?
// TODO create proc_macro crate
// TODO grade sub, heterogeneous add/sub,reverses, complements, antiproducts
// Geometric algebra feature set:
//   https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

pub trait Commutator<Rhs> {
    type Output;
    fn commutator(self, rhs: Rhs) -> Self::Output;
}

pub trait Wedge<Rhs> {
    type Output;
    fn wedge(self, rhs: Rhs) -> Self::Output;
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Zero;

impl<T> std::ops::Add<T> for Zero {
    type Output = T;
    fn add(self, rhs: T) -> T {
        rhs
    }
}

impl<T: std::ops::Neg<Output = T>> std::ops::Sub<T> for Zero {
    type Output = T;
    fn sub(self, rhs: T) -> T {
        -rhs
    }
}

impl<T> std::ops::Mul<T> for Zero {
    type Output = Zero;
    fn mul(self, _rhs: T) -> Zero {
        Zero
    }
}

impl const From<Zero> for f64 {
    fn from(_: Zero) -> f64 {
        0.0
    }
}

#[cfg(feature = "pga_3d")]
pub mod pga_3d {
    proc_macros::clifford!(3, 0, 1);

    pub const fn point(x: f64, y: f64, z: f64) -> Vector {
        Vector::new(x, y, z, 1.)
    }

    #[test]
    fn vec_mul() {
        let a = Vector::new(2., 3., 4., 1.);
        let b = Vector::new(3., 5., 7., 1.);
        let c = Vector::new(0., 0., 0., 1.);

        let line = a.wedge(b);
        let plane = line.wedge(c);

        dbg!(line, plane, a * b);
        // panic!("done");
    }
}

#[cfg(feature = "cga_2d")]
pub mod cga_2d {
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
        T: Wedge<Vector, Output = U>,
        U: From<Zero> + PartialEq,
    {
        fn is_flat(self) -> bool {
            self.wedge(N_BAR) == U::from(Zero)
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
}

#[cfg(feature = "cga_3d")]
pub mod cga_3d {
    proc_macros::clifford!(4, 1, 0);

    /// Point at the origin
    pub const N: Vector = Vector::new(0., 0., 0., 0.5, 0.5);

    /// Point through infinity
    pub const N_BAR: Vector = Vector::new(0., 0., 0., -1., 1.);

    pub const fn point(x: f64, y: f64, z: f64) -> Vector {
        let x2 = x * x + y * y + z * z;
        Vector::new(x, y, z, 0.5 - 0.5 * x2, 0.5 + 0.5 * x2)
    }

    #[test]
    fn vector_test() {
        let x = Vector::new(2., 3., 5., 0., 0.);
        let x2 = 2. * 2. + 3. * 3. + 5. * 5.;
        let expected = x + 0.5 * x2 * N_BAR + N;

        let actual = point(2., 3., 5.);

        assert_eq!(expected, actual);
    }

    trait IsFlat {
        fn is_flat(self) -> bool;
    }

    impl<T, U> IsFlat for T
    where
        T: Wedge<Vector, Output = U>,
        U: From<Zero> + PartialEq,
    {
        fn is_flat(self) -> bool {
            self.wedge(N_BAR) == U::from(Zero)
        }
    }

    #[test]
    fn flat_test() {
        let o = point(0., 0., 0.);
        let a = point(1., 0., 0.);
        let b = point(0., 1., 0.);
        let c = point(0., 0., 1.);

        let points: Bivector = o.wedge(a);
        let line: Trivector = points.wedge(N_BAR);
        let circle: Trivector = points.wedge(b);
        let plane: Quadvector = line.wedge(c);
        let sphere: Quadvector = circle.wedge(c);

        assert!(!o.is_flat());
        assert!(!points.is_flat());
        assert!(!circle.is_flat());
        assert!(!sphere.is_flat());
        assert!(line.is_flat());
        assert!(plane.is_flat());
    }
}
