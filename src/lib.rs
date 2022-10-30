// IDEA if I cant impl Mul<*> for T, I can convert all to Mul<T> for *
//   scalar ops are all commutative

// TODO Multivector type parameters should be the inner type
// TODO replace Zero with f0 to prevent collisions with num_traits::Zero

// TODO figure out where impl<T: Float> Trait for T can be used instead of "for f32" and "for f64"
//   lets us use any Float type, including num-dual
//   Option 1 :: Scalar<T>
//     allows standard operators but requires wrapping the value
//     problem: still can't define generic ops for generic types and f0/Zero
//   Option 2 :: Mul: BinaryOp<Lhs, Rhs>
//     prevents using the standard operators, but allows generics as the Lhs type
//   Option 3 :: impl Mul only for scalar multiplication, Geo/Dot/Wedge for geometric transformations

// TODO Unit -> UnitOp, add Unit<T> with Norm and Norm2 == 1

// TODO impl num_traits::Zero for types
// TODO scalar ops (Mul<T>, Add<T>, Sub<T>) need T: num_traits::Float to compile

// TODO use complement to find blades that need to be reversed (e.g., e12, e23, e31)?
//  - is this unique to G{3,0,1} ?
//  - allow manual blade ordering

//! Proc macros for defining Clifford algebras of arbitrary dimension
//!
//! [`Feature set`]
//!
//! Models of geometry:
//! - [x] Generic types
//!     - [x] f32/f64 conversions
//!     - [x] flexible generics
//! - [ ] Euclidean
//!     - [ ] Meet
//!     - [ ] Join
//! - [ ] Homogeneous
//!     - [ ] Weight
//!     - [ ] Bulk
//!     - [ ] IsIdeal
//!     - [ ] Projection
//!     - [ ] Antiprojection
//! - [ ] Conformal
//!     - [ ] Meet
//!     - [ ] Join
//!     - [ ] Origin
//!     - [ ] Infinity
//!     - [ ] IsFlat
//! - [ ] Minkowski
//!
//! Types:
//! - [x] Grades
//! - [x] Unit
//!
//! Functions:
//! - [x] Grade selection (e.g., Motor::bivector() -> Bivector)
//! - [x] fn new(...) -> Self { ... }
//! - [x] From<Grade> for Versor/Multivector
//!
//! Main products:
//! - [x] Mul scalar
//! - [x] Div scalar
//! - [x] Geometric
//! - [x] Antigeometric
//! - [x] Grade products
//! - [x] Grade antiproducts
//!
//! Inner products:
//! - [x] Dot
//! - [x] Antidot
//! - [ ] Left contraction
//! - [ ] Right contraction
//!
//! Outer products:
//! - [x] Wedge
//! - [x] Antiwedge (regressive)
//!
//! Interior products:
//! - [ ] Right interior product
//! - [ ] Left interior product
//! - [ ] Right interior antiproduct
//! - [ ] Left interior antiproduct
//!
//! Sum products:
//! - [x] Addition
//! - [x] Subtraction
//!
//! Assignment:
//! - [x] AddAssign
//! - [x] SubAssign
//! - [x] MulAssign scalar
//! - [x] DivAssign scalar
//!
//! Unary operations:
//! - [x] Neg
//! - [x] Left complement
//! - [x] Right complement
//! - [x] Reverse
//! - [x] Antireverse
//!
//! Norm-based operations:
//! - [x] Norm
//! - [x] Norm2
//! - [x] Inverse
//! - [x] Unitize
//!
//! Compound products:
//! - [x] Sandwich
//! - [x] Antisandwich
//! - [ ] Commutator
//!
//! Operator overloading:
//! - [ ] Dot product: a | b
//! - [ ] Wedge product: a ^ b
//! - [ ] Antiwedge (regressive) product: a & b
//!
//! Num Traits:
//! - [x] Zero
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub use proc_macros::clifford;

#[cfg(feature = "ga_3d")]
pub mod ga_3d {
    macros::g3!();

    #[test]
    fn value_geo() {
        let v = Value::Vector(Vector::new(1., 0., 0.));
        assert_eq!(
            Value::Motor(Motor {
                s: 1.,
                ..Motor::default()
            }),
            v.geo(v).unwrap()
        );
    }
}

#[cfg(feature = "pga_3d")]
pub mod pga3 {
    macros::pga3!();

    impl<T> num_sqrt::Sqrt for Motor<T>
    where
        T: num_sqrt::Sqrt<Output = T> + num_traits::One,
        Motor<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>
            + Norm2<Output = Scalar<T>>
            + Unitize<Output = Unit<Motor<T>>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            let sqrt = self.unit().value() + Scalar { s: T::one() };
            sqrt.unit().value()
        }
    }

    #[test]
    fn rotor_sqrt() {
        use num_sqrt::Sqrt;

        let rot_180 = Motor {
            s: 0.,
            xy: -1.,
            ..Default::default()
        };
        let v = Vector {
            x: 2.,
            y: 3.,
            z: 5.,
            w: 1.,
        };
        let v_90 = Vector {
            x: -3.,
            y: 2.,
            z: 5.,
            w: 1.,
        };

        let rot_90 = rot_180.sqrt();
        dbg!(rot_90, rot_90.sandwich(v));

        assert_eq!(v_90, rot_90.sandwich(v));
        // panic!("done");
    }

    #[test]
    fn motor_from_scalar() {
        let s = Scalar { s: 1. };
        let _m = Motor::from(s);
    }

    #[test]
    fn rotation_and_unit_rotation() {
        let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
        let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.) * 2.;
        let unit = motor.unit();

        let v = Vector::new(1., 2., 3., 0.);

        let v1 = motor.sandwich(v);
        let v2 = unit.sandwich(v); // interestingly, the sqrt adds inaccuracy

        dbg!(v1, v2);
        assert!((v1 - v2).norm2().s < 1e-10);
        // panic!();
    }

    #[test]
    fn f0_ops() {
        use f_zero::f0;
        let vector: Value<f0> = Value::Vector(Default::default());
        let bivector: Value<f0> = Value::Bivector(Default::default());
        assert_eq!(
            Some(Value::Flector(Default::default())),
            vector.geo(bivector)
        );
        assert_eq!(
            Some(Value::Vector(Default::default())),
            vector.dot(bivector)
        );
    }
}
