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
//! - [ ] Homogeneous
//!     - [x] Weight
//!     - [x] Bulk
//!     - [ ] IsIdeal
//!     - [ ] Projection
//!     - [ ] Antiprojection
//! - [ ] Conformal
//!     - [ ] Meet
//!     - [ ] Join
//!     - [x] Origin
//!     - [x] Infinity
//!     - [x] IsFlat
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
//! - [x] Antigeometric
//!
//! Inner products:
//! - [x] Dot
//! - [x] Antidot
//! - [x] Left contraction
//! - [x] Right contraction
//!
//! Outer products:
//! - [x] Wedge
//! - [x] Antiwedge
//!
//! Sum products:
//! - [x] Addition
//! - [x] Subtraction
//!
//! Assignment:
//! - [ ] Add/SubAssign
//! - [ ] Mul/DivAssign
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
//! - [ ] Antinorm
//! - [ ] Antinorm2
//! - [ ] AntiInverse
//! - [ ] AntiUnit
//!
//! Compound products:
//! - [ ] Sandwich
//! - [ ] Antisandwich
//! - [ ] Commutator
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub use proc_macros::clifford;

#[cfg(feature = "ga_3d")]
pub mod ga_3d;

#[cfg(feature = "pga_3d")]
pub mod pga_3d;

#[cfg(feature = "cga_2d")]
pub mod cga_2d;

#[cfg(feature = "cga_3d")]
pub mod cga_3d;
