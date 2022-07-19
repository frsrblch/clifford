#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert
)]
// TODO use complement to find blades that need to be reversed (e.g., e12, e23, e31)?
//  - is this unique to G{3,0,1} ?
//  - allow manual blade ordering

// TODO compare plane-based and point-based algebras,
//  - can objects in one be converted to the other through the dual?
//      - point-based point: x e1 + y e2 + z e3 + 1 e4
//      - plane-based plane: x e1 + y e2 + z e3 + δ e4

//! Proc macros for defining Clifford algebras of arbitrary dimension
//!
//! [`Feature set`]
//!
//! Models of geometry:
//! - [ ] Generic types
//!     - [ ] f32/f64 conversions
//!     - [ ] dimensional analysis
//! - [ ] Euclidean
//!     - [ ] Meet
//!     - [ ] Join
//! - [ ] Homogeneous 3D - points as vectors
//!     - [x] Antigeometric
//!     - [x] Antiwedge
//!     - [x] Antidot
//!     - [ ] Antireverse
//!     - [ ] Meet
//!     - [ ] Join
//!     - [x] Weight
//!     - [x] Bulk
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
//! - [x] Left contraction
//! - [x] Right contraction
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
//! - [ ] Antireverse
//!
//! Norm-based operations:
//! - [x] Norm
//! - [x] NormSquared
//! - [x] Inverse
//! - [x] Unitize
//!
//! Multivector operations:
//! - [ ] PartialEq<Grade>, asserts that other grades are zero
//!
//! Compound products:
//! - [ ] Sandwich
//! - [ ] Antisandwich
//! - [ ] Commutator
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub mod va_3d_manual;

#[cfg(feature = "ga_3d")]
pub mod ga_3d;

#[cfg(feature = "va_3d_mv")]
pub mod va_3d_mv;

pub use proc_macros::clifford;
