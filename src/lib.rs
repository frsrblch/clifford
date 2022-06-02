#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert
)]
// TODO add efficient implementation for null cone geometries

//! Proc macros for defining Clifford algebras of arbitrary dimension
//!
//! [`Feature set`]
//!
//! Models of geometry:
//! - [ ] Euclidean
//!     - [ ] Meet
//!     - [ ] Join
//! - [ ] Homogeneous 3D - points as vectors
//!     - [ ] Antigeometric
//!     - [ ] Antiwedge
//!     - [ ] Antidot
//!     - [ ] Antireverse
//!     - [ ] Meet
//!     - [ ] Join
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
//! - [x] Zero
//! - [x] Grades
//! - [x] Multivector
//!
//! Main products:
//! - [x] Mul
//! - [x] Div
//! - [x] Geometric
//! - [ ] Commutator
//! - [ ] Sandwich
//!
//! Inner products:
//! - [x] Dot
//! - [ ] Left contraction
//! - [ ] Right contraction
//!
//! Outer products:
//! - [x] Wedge
//!
//! Sum products:
//! - [x] Addition
//! - [x] Subtraction
//!
//! Unary operations:
//! - [x] Neg
//! - [x] Left complement
//! - [x] Right complement
//! - [x] Reverse
//!
//! Norm-based operations:
//! - [ ] Inverse
//! - [ ] Normalize
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

// pub mod va_3d_manual;
//
// #[cfg(feature = "va_3d_mv")]
// pub mod va_3d_mv;

// #[cfg(feature = "va_3d")]
// pub mod va_3d;
//
// #[cfg(feature = "pga_3d")]
// pub mod pga_3d;
//
// #[cfg(feature = "cga_2d")]
// pub mod cga_2d;

#[cfg(feature = "cga_3d")]
pub mod cga_3d;

pub use proc_macros::clifford;

pub trait GradeAdd<Rhs> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

pub trait GradeSub<Rhs> {
    type Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
}

pub trait Geometric<Rhs> {
    type Output;
    fn geo(self, rhs: Rhs) -> Self::Output;
}

// impl Geometric<f64> for f64 {
//     type Output = f64;
//     fn geo(self, rhs: Self) -> Self {
//         self * rhs
//     }
// }

pub trait Wedge<Rhs> {
    type Output;
    fn wedge(self, rhs: Rhs) -> Self::Output;
}

// impl Wedge<f64> for f64 {
//     type Output = f64;
//     fn wedge(self, rhs: Self) -> Self {
//         self * rhs
//     }
// }

pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

// impl Dot<f64> for f64 {
//     type Output = f64;
//     fn dot(self, rhs: Self) -> Self {
//         self * rhs
//     }
// }

pub trait Sandwich<Rhs> {
    type Output;
    fn sandwich(self, rhs: Rhs) -> Self::Output;
}

// impl Sandwich<f64> for f64 {
//     type Output = f64;
//
//     fn sandwich(self, rhs: f64) -> Self::Output {
//         rhs
//     }
// }

pub trait Commutator<Rhs> {
    type Output;
    fn commutator(self, rhs: Rhs) -> Self::Output;
}

pub trait Reverse {
    type Output;
    fn rev(self) -> Self::Output;
}

// impl Reverse for f64 {
//     type Output = f64;
//     fn rev(self) -> Self {
//         self
//     }
// }

pub trait LeftComplement {
    type Output;
    fn left_comp(self) -> Self::Output;
}

pub trait RightComplement {
    type Output;
    fn right_comp(self) -> Self::Output;
}

pub trait Bulk {
    type Output;
    fn bulk(self) -> Self::Output;
}

pub trait Weight {
    type Output;
    fn weight(self) -> Self::Output;
}

pub trait Antireverse {
    type Output;
    fn antirev(self) -> Self::Output;
}

impl<T, Comp, CompRev> Antireverse for T
where
    T: LeftComplement<Output = Comp>,
    Comp: Reverse<Output = CompRev>,
    CompRev: RightComplement,
{
    type Output = CompRev::Output;

    #[inline]
    fn antirev(self) -> Self::Output {
        self.left_comp().rev().right_comp()
    }
}

pub trait Antigeometric<Rhs> {
    type Output;
    fn antigeo(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antigeometric<Rhs> for Lhs
where
    Lhs: LeftComplement<Output = LhsComp>,
    Rhs: LeftComplement<Output = RhsComp>,
    LhsComp: Geometric<RhsComp, Output = OutputComp>,
    OutputComp: RightComplement,
{
    type Output = OutputComp::Output;
    #[inline]
    fn antigeo(self, rhs: Rhs) -> Self::Output {
        let lhs = self.left_comp();
        let rhs = rhs.left_comp();
        let output_complement = lhs.geo(rhs);
        output_complement.right_comp()
    }
}

pub trait Antiwedge<Rhs> {
    type Output;
    fn antiwedge(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antiwedge<Rhs> for Lhs
where
    Lhs: LeftComplement<Output = LhsComp>,
    Rhs: LeftComplement<Output = RhsComp>,
    LhsComp: Wedge<RhsComp, Output = OutputComp>,
    OutputComp: RightComplement,
{
    type Output = OutputComp::Output;

    #[inline]
    fn antiwedge(self, rhs: Rhs) -> Self::Output {
        let lhs = self.left_comp();
        let rhs = rhs.left_comp();
        let output_complement = lhs.wedge(rhs);
        output_complement.right_comp()
    }
}

pub trait Antidot<Rhs> {
    type Output;
    fn antidot(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antidot<Rhs> for Lhs
where
    Lhs: LeftComplement<Output = LhsComp>,
    Rhs: LeftComplement<Output = RhsComp>,
    LhsComp: Dot<RhsComp, Output = OutputComp>,
    OutputComp: RightComplement,
{
    type Output = OutputComp::Output;

    #[inline]
    fn antidot(self, rhs: Rhs) -> Self::Output {
        let lhs = self.left_comp();
        let rhs = rhs.left_comp();
        let output_complement = lhs.dot(rhs);
        output_complement.right_comp()
    }
}

pub trait Antisandwich<Rhs> {
    type Output;
    fn antisandwich(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antisandwich<Rhs> for Lhs
where
    Lhs: LeftComplement<Output = LhsComp>,
    Rhs: LeftComplement<Output = RhsComp>,
    LhsComp: Sandwich<RhsComp, Output = OutputComp>,
    OutputComp: RightComplement,
{
    type Output = OutputComp::Output;
    #[inline]
    fn antisandwich(self, rhs: Rhs) -> Self::Output {
        let lhs = self.left_comp();
        let rhs = rhs.left_comp();
        let output_complement = lhs.sandwich(rhs);
        output_complement.right_comp()
    }
}

pub trait IsIdeal {
    fn is_ideal(&self) -> bool;
}

impl<T> IsIdeal for T
where
    T: PartialEq<T::Output> + Bulk + Copy,
{
    #[inline]
    fn is_ideal(&self) -> bool {
        self.eq(&self.bulk())
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Zero;

impl const std::ops::Add for Zero {
    type Output = Zero;
    #[inline]
    fn add(self, _: Self) -> Self {
        Zero
    }
}

impl const std::ops::Sub for Zero {
    type Output = Zero;
    #[inline]
    fn sub(self, _: Self) -> Self {
        Zero
    }
}
