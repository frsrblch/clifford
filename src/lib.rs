//! Proc macros for defining geometric algebras of arbitrary dimension and basis vectors
//!
//! [`Feature set`]
//!
//! Types:
//! - [x] Grades
//! - [x] Even-grade
//! - [x] Odd-grade
//! - [x] Multivector
//! - [x] Unit typestate
//!     - [x] Seamless use of unary and binary operations
//!
//! Generics:
//! - [x] flexible generics (e.g., `Vector<T>: Mul<Scalar<U>, Output = Vector<V>> where T: Mul<U, Output = V>`)
//! - [x] FloatType trait (e.g., `Vector<T>: FloatType<Float = T>`)
//!
//! Functions:
//! - [x] Grade selection (e.g., `Motor::bivector() -> Bivector`)
//! - [x] fn new(...) -> Self { ... }
//! - [x] `From<Grade>` for `Motor`/`Flector`/`Multivector`
//!
//! Main products:
//! - [x] Mul
//! - [x] Div
//! - [x] Geometric
//! - [x] Antigeometric
//! - [x] Grade products
//! - [x] Grade antiproducts
//! - [x] Commutator
//!
//! Inner products:
//! - [x] Dot
//! - [x] Antidot
//! - [x] Left contraction
//! - [x] Right contraction
//!
//! Outer products:
//! - [x] Wedge
//! - [x] Antiwedge (regressive)
//!
//! Sum products:
//! - [x] Addition
//! - [x] Subtraction
//! - [x] Add/Sub f32/f64
//!
//! Assignment:
//! - [x] AddAssign
//! - [x] SubAssign
//! - [x] MulAssign
//! - [x] DivAssign
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
//! - [x] Dot product: `a | b`
//! - [x] Wedge product: `a ^ b`
//! - [x] Antiwedge (regressive) product: `a & b`
//! - [x] Sandwich product: `M >> a`
//! - [x] Dual: `!a`
//!
//! Num Traits:
//! - [x] Zero
//! - [x] One
//! - [x] Float for `Scalar<T>`
//!
//! Rand:
//! - [x] `Unit<T>` for r-vectors where 0 < r < n
//! - [x] Random r-vector with norm <= 1
//! - [x] Random r-vector with norm == 1
//!
//! Iter:
//! - [x] std::iter::Sum
//! - [x] std::iter::Product
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.

pub use macros::algebra;