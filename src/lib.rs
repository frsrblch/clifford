//! Proc macros for defining geometric algebras of arbitrary dimension and basis vectors
//!
//! [`Feature set`]
//!
//! Models of geometry:
//! - [x] Vectorspace GA 3D
//! - [x] Homogeneous PGA 3D
//! - [ ] Conformal CGA 3D (experimental WIP)
//!     - [x] Origin
//!     - [x] Infinity
//!     - [x] IsFlat
//! - [ ] Minkowski
//!
//! Types:
//! - [x] Grades
//! - [x] Even-grade
//! - [x] Odd-grade
//! - [x] Multivector
//! - [x] Unit
//!     - [ ] Seamless use of unary and binary operations with `Unit<T>` types
//!
//! Generics:
//! - [x] flexible generics (e.g., `Vector<T>: Mul<Scalar<U>, Output = Vector<V>> where T: Mul<U, Output = V>`)
//! - [x] f32/f64 conversions
//! - [x] FloatType trait (e.g., `Vector<T>: FloatType<Float = T>`)
//!
//! Functions:
//! - [x] Grade selection (e.g., `Motor::bivector() -> Bivector`)
//! - [x] fn new(...) -> Self { ... }
//! - [x] `From<Grade>` for `Versor`/`Multivector`
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
//! - [x] Add/Sub f32/f64
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
//! - [x] Random r-vector with norm < 1
//!
//! [`Feature set`]: https://ga-developers.github.io/ga-benchmark-runs/2020.02.05/table_of_features.html

pub use macros::{algebra, algebra_slim};
