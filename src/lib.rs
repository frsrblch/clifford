#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert,
    core_intrinsics
)]
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
//!
//! Functions:
//! - [x] Grade selection (e.g., Motor::bivector() -> Bivector)
//!
//! Main products:
//! - [x] Mul scalar
//! - [x] Div scalar
//! - [x] Geometric
//! - [ ] Antigeometric
//! - [x] Grade products
//! - [ ] Grade antiproducts
//!
//! Inner products:
//! - [x] Dot
//! - [ ] Antidot
//! - [ ] Left contraction
//! - [ ] Right contraction
//!
//! Outer products:
//! - [x] Wedge
//! - [ ] Antiwedge
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
//! - [ ] Antireverse
//!
//! Norm-based operations:
//! - [x] Norm
//! - [x] Norm2
//! - [x] Inverse
//! - [x] Unitize
//!
//! Compound products:
//! - [x] Sandwich
//! - [ ] Antisandwich
//! - [ ] Commutator
//!
//! Num Traits:
//! - [x] Zero
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub use proc_macros::clifford;

#[cfg(feature = "ga_3d")]
pub mod ga_3d;

pub mod pga3 {
    macros::pga3!();

    pub fn unit_sandwich(motor: Motor<f64>, vector: Vector<f64>) -> Vector<f64> {
        let intermediate: Flector<f64> = Geo::<Vector<f64>>::geo(motor, vector);
        <Vector<f64> as GradeProduct<Flector<f64>, Motor<f64>>>::product(
            intermediate,
            Reverse::rev(motor),
        )
    }

    impl<Lhs, Rhs, Int> Sandwich<Rhs> for Unit<Lhs>
    where
        Lhs: Geo<Rhs, Output = Int> + Reverse + Copy,
        Rhs: GradeProduct<Int, Lhs>,
    {
        type Output = Rhs;
        #[inline]
        fn sandwich(self, rhs: Rhs) -> Self::Output {
            let int = self.value().geo(rhs);
            Rhs::product(int, self.value().rev())
        }
    }

    impl<T> num_sqrt::Sqrt for Motor<T>
    where
        T: num_sqrt::Sqrt<Output = T> + num_traits::One,
        Motor<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>
            + Norm2<Output = T>
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
        let motor = Motor {
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
        let v_ = Vector {
            x: -3.,
            y: 2.,
            z: 5.,
            w: 1.,
        };

        use num_sqrt::Sqrt;
        let sqrt = motor.sqrt();
        let sqrt2 = sqrt.sqrt();
        dbg!(sqrt, sqrt.sandwich(v));
        dbg!(sqrt2, sqrt2.sandwich(v));

        assert_eq!(v_, sqrt.sandwich(v));
        // panic!("done");
    }
}

// #[cfg(feature = "pga_3d")]
// pub mod pga_3d;

// #[cfg(feature = "cga_2d")]
// pub mod cga_2d;

#[cfg(feature = "cga_3d")]
pub mod cga_3d;
