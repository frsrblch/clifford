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
//! - [x] Commutator
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
//!
//! Operator overloading:
//! - [x] Dot product: a | b
//! - [x] Wedge product: a ^ b
//! - [x] Antiwedge (regressive) product: a & b
//!
//! Num Traits:
//! - [x] Zero
//! - [x] One
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub use proc_macros::clifford;

#[cfg(feature = "ga_3d")]
pub mod ga_3d {
    macros::ga3!();

    #[cfg(feature = "dyn")]
    macros::dyn_ga3!();

    impl<T> num_sqrt::Sqrt for Bivector<T>
    where
        Unit<Bivector<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
        Bivector<T>: Unitize<Output = Unit<Bivector<T>>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.unit().sqrt()
        }
    }

    impl<T> num_sqrt::Sqrt for Unit<Bivector<T>>
    where
        Scalar<T>: num_traits::One,
        Bivector<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.value() + num_traits::One::one()
        }
    }

    impl<T> num_sqrt::Sqrt for Motor<T>
    where
        Motor<T>: Unitize<Output = Unit<Motor<T>>>,
        Unit<Motor<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.unit().sqrt()
        }
    }

    impl<T> num_sqrt::Sqrt for Unit<Motor<T>>
    where
        Scalar<T>: num_traits::One,
        Motor<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.value() + num_traits::One::one()
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Vector<T>>> for rand::distributions::Standard
    where
        T: num_traits::One
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + PartialOrd
            + Copy,
        Vector<T>: Norm2<Output = Scalar<T>> + Unitize<Output = Unit<Vector<T>>>,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Vector<T>> {
            loop {
                let two = T::one() + T::one();

                let x = rng.gen::<T>() * two - T::one();
                let y = rng.gen::<T>() * two - T::one();
                let z = rng.gen::<T>() * two - T::one();

                let v = Vector { x, y, z };

                if v.norm2().s <= T::one() {
                    return v.unit();
                }
            }
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Bivector<T>>> for rand::distributions::Standard
    where
        rand::distributions::Standard: rand::distributions::Distribution<Unit<Vector<T>>>,
        Vector<T>: Dual<Output = Bivector<T>>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Bivector<T>> {
            Unit(rng.gen::<Unit<Vector<T>>>().value().dual())
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
    where
        rand::distributions::Standard: rand::distributions::Distribution<Unit<Bivector<T>>>
            + rand::distributions::Distribution<T>,
        T: From<f64>
            + std::ops::Mul<Output = T>
            + num_trig::Sin<Output = T>
            + num_trig::Cos<Output = T>
            + Copy,
        Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
            + std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
            let bivector = rng.gen::<Unit<Bivector<T>>>();
            let angle = rng.gen::<T>() * T::from(std::f64::consts::PI);
            let sin = Scalar { s: angle.sin() };
            let cos = Scalar { s: angle.cos() };
            Unit(bivector.value() * cos + sin)
        }
    }

    #[cfg(test)]
    mod rand_tests {
        use rand::{thread_rng, Rng};

        use super::*;

        #[test]
        fn unit_vector() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let v = rng.gen::<Unit<Vector<f64>>>();
                assert_eq!(v.value().norm2().to_f32().s, 1.);
            }
        }

        #[test]
        fn unit_bivector() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let b = rng.gen::<Unit<Bivector<f64>>>();
                assert_eq!(b.value().norm2().to_f32().s, 1.);
            }
        }

        #[test]
        fn unit_motor() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let m = rng.gen::<Unit<Motor<f64>>>();
                assert_eq!(m.value().norm2().to_f32().s, 1.);
            }
        }
    }
}

#[cfg(feature = "pga_3d")]
pub mod pga3 {
    macros::pga3!();

    #[cfg(feature = "dyn")]
    macros::dyn_pga3!();

    impl<T> num_sqrt::Sqrt for Bivector<T>
    where
        Unit<Bivector<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
        Bivector<T>: Unitize<Output = Unit<Bivector<T>>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.unit().sqrt()
        }
    }

    impl<T> num_sqrt::Sqrt for Unit<Bivector<T>>
    where
        Scalar<T>: num_traits::One,
        Bivector<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.value() + num_traits::One::one()
        }
    }

    impl<T> num_sqrt::Sqrt for Motor<T>
    where
        Motor<T>: Unitize<Output = Unit<Motor<T>>>,
        Unit<Motor<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.unit().sqrt()
        }
    }

    impl<T> num_sqrt::Sqrt for Unit<Motor<T>>
    where
        Scalar<T>: num_traits::One,
        Motor<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        type Output = Motor<T>;

        fn sqrt(self) -> Self::Output {
            self.value() + num_traits::One::one()
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Vector<T>>> for rand::distributions::Standard
    where
        T: num_traits::One
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + PartialOrd
            + Copy,
        Vector<T>: Norm2<Output = Scalar<T>> + Unitize<Output = Unit<Vector<T>>>,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Vector<T>> {
            loop {
                let two = T::one() + T::one();

                let x = rng.gen::<T>() * two - T::one();
                let y = rng.gen::<T>() * two - T::one();
                let z = rng.gen::<T>() * two - T::one();

                let v = Vector {
                    x,
                    y,
                    z,
                    w: T::zero(),
                };

                if v.norm2().s <= T::one() {
                    return v.unit();
                }
            }
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Bivector<T>>> for rand::distributions::Standard
    where
        rand::distributions::Standard: rand::distributions::Distribution<Unit<Vector<T>>>,
        T: num_traits::Zero,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Bivector<T>> {
            let v = rng.gen::<Unit<Vector<T>>>().value();
            Unit(Bivector {
                xy: v.z,
                yz: v.x,
                xz: v.y,
                ..num_traits::Zero::zero()
            })
        }
    }

    impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
    where
        rand::distributions::Standard: rand::distributions::Distribution<Unit<Bivector<T>>>
            + rand::distributions::Distribution<T>,
        T: From<f64>
            + std::ops::Mul<Output = T>
            + num_trig::Sin<Output = T>
            + num_trig::Cos<Output = T>
            + Copy,
        Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
            + std::ops::Add<Scalar<T>, Output = Motor<T>>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
            let bivector = rng.gen::<Unit<Bivector<T>>>();
            let angle = rng.gen::<T>() * T::from(std::f64::consts::PI);
            let sin = Scalar { s: angle.sin() };
            let cos = Scalar { s: angle.cos() };
            Unit(bivector.value() * cos + sin)
        }
    }

    #[cfg(test)]
    mod rand_tests {
        use rand::{thread_rng, Rng};

        use super::*;

        #[test]
        fn unit_vector() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let v = rng.gen::<Unit<Vector<f64>>>();
                assert_eq!(v.value().norm2().to_f32().s, 1.);
            }
        }

        #[test]
        fn unit_bivector() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let b = rng.gen::<Unit<Bivector<f64>>>();
                assert_eq!(b.value().norm2().to_f32().s, 1.);
            }
        }

        #[test]
        fn unit_motor() {
            let mut rng = thread_rng();
            for _ in 0..100 {
                let m = rng.gen::<Unit<Motor<f64>>>();
                assert_eq!(m.value().norm2().to_f32().s, 1.);
            }
        }
    }
}
