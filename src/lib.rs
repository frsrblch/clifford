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

pub trait Join<Rhs> {
    type Output;
    fn join(self, rhs: Rhs) -> Self::Output;
}

pub trait Meet<Rhs> {
    type Output;
    fn meet(self, rhs: Rhs) -> Self::Output;
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

#[cfg(feature = "va_3d")]
pub mod va_3d {
    proc_macros::clifford!(3, 0, 0);

    #[test]
    fn vector_autodif_test() {
        let x = Vector::new(1., 2., 3.);
        let v = Vector::new(-5., 7., 11.);

        let ux = x / x.dot(x).sqrt();
        let uv = v / v.dot(v).sqrt();

        let p: Even = ux * uv;
        dbg!(p);

        // panic!("{:?}", p);
    }

    #[test]
    fn accel_derivative() {
        let a = |x: Vector| -x / x.dot(x).powf(1.5);
        let x_0 = Vector::new(0., 0., 2.);

        let norm = |v: Vector| v.dot(v).sqrt();

        // let dx = Vector::new(1e-6, 0., 0.);
        // let dy = Vector::new(0., 1e-6, 0.);
        // let dz = Vector::new(0., 0., 1e-6);

        let da = |x: Vector| {
            let inv_det = x.dot(x).powf(-2.5);
            let Vector {
                e1: x,
                e2: y,
                e3: z,
            } = x;
            let (x2, y2, z2) = (x * x, y * y, z * z);
            let da = Vector::new(
                -2. * x2 + y2 + z2 - 3. * x * (y + z),
                x2 - 2. * y2 + z2 - 3. * y * (x + z),
                x2 + y2 - 2. * z2 - 3. * z * (x + y),
            ) * inv_det;
            move |h: Vector| Vector::new(da.e1 * h.e1, da.e2 * h.e2, da.e3 * h.e3) / norm(h)
        };

        // dbg!(da(x_0)(dx));
        // dbg!(da(x_0)(dy));
        // dbg!(da(x_0)(dz));

        let dt = 0.05;
        let dt2 = dt * dt;
        let mut x = x_0;
        let mut v = Vector::new(0., 0.01, 0.02);

        dbg!(x, v);
        for _ in 0..20 {
            let a = a(x) + 0.5 * da(x)(v * dt);

            x = x + dt * v + 0.5 * dt2 * a;
            v = v + dt * a;

            dbg!(x, v);
        }

        // panic!("done");
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
        let a = point(2., 3., 4.);
        let b = point(3., 5., 7.);
        let c = point(0., 0., 0.);

        let line = a.wedge(b);
        let plane = line.wedge(c);

        dbg!(line, plane, a * b);

        // panic!("done");
    }

    #[test]
    fn plane_dual() {
        let o = point(0., 0., 0.);
        let x = point(1., 0., 0.);
        let y = point(0., 1., 0.);

        let plane = o.wedge(x).wedge(y);
        dbg!(plane);

        // panic!();
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
        T: crate::Wedge<Vector, Output = U>,
        U: From<crate::Zero> + PartialEq,
    {
        fn is_flat(self) -> bool {
            self.wedge(N_BAR) == U::from(crate::Zero)
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
        T: crate::Wedge<Vector, Output = U>,
        U: From<crate::Zero> + PartialEq,
    {
        fn is_flat(self) -> bool {
            self.wedge(N_BAR) == U::from(crate::Zero)
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
