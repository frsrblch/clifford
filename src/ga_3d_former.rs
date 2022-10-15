// proc_macros::clifford!(3, 0, 0, [x, y, z]);

use num_traits::Float;

pub type Motor<T> = Multivector<T, Zero, Bivector<T>, Zero>;

impl<T: Float + std::fmt::Display> Motor<T> {
    pub fn sqrt(&self) -> Self {
        self.debug_assert_unit();

        let half = T::one() / (T::one() + T::one());
        let angle: T = self.0.acos() * half;
        let (sin, cos) = angle.sin_cos();

        if let Some(bivector) = self.2.try_unit() {
            Multivector(cos, Zero, bivector * sin, Zero)
        } else {
            Multivector(cos, Zero, num_traits::Zero::zero(), Zero)
        }
    }

    /// (PI - angle) is a half rotation in the opposite direction
    pub fn angle_axis(self) -> (T, Vector<T>) {
        // multiply by two because the sandwich operator uses the motor twice
        let two = T::one() + T::one();
        self.debug_assert_unit();
        (
            self.0.acos() * two,
            self.2
                .try_unit()
                .unwrap_or_else(|| Bivector::new(T::zero(), T::zero(), T::zero()))
                .dual(),
        )
    }

    pub fn from_axis_angle(axis: Vector<T>, angle: T) -> Self {
        let half = T::one() / (T::one() + T::one());
        let bivector = axis.dual();
        let (sin, cos) = (angle * half).sin_cos();
        Multivector(cos, Zero, bivector * sin, Zero)
    }

    fn debug_assert_unit(&self) {
        let norm2 = self.0 * self.0 + self.2.norm2();
        let t: T = norm2.sub(T::one()).abs();
        debug_assert!(
            t < T::epsilon().sqrt(),
            "expected unit motor: norm^2 = {}",
            norm2
        );
    }
}

impl<T: Float> Bivector<T> {
    #[inline]
    pub fn try_unit(self) -> Option<Self> {
        let norm = self.norm();
        if norm == T::zero() {
            None
        } else {
            Some(self / norm)
        }
    }
}

impl<T: Float> num_traits::Zero for Bivector<T> {
    fn zero() -> Self {
        Bivector::new(T::zero(), T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.xy.is_zero() & self.xz.is_zero() & self.xy.is_zero()
    }
}

macro_rules! impl_op_assign {
    ($type_:ident { $($field:ident),* }) => {
        impl_op_assign!($type_ { $($field),*} MulAssign :: mul_assign);
        impl_op_assign!($type_ { $($field),*} DivAssign :: div_assign);
    };
    ($type_:ident { $($field:ident),* }  $trait_:ident :: $fn_:ident) => {
        impl<T: std::ops::$trait_<T> + Copy> std::ops::$trait_<T> for $type_<T> {
            fn $fn_(&mut self, rhs: T) {
                $( std::ops::$trait_::$fn_(&mut self.$field, rhs); )*
            }
        }

        impl<T: std::ops::$trait_<T> + Copy> std::ops::$trait_<&T> for $type_<T> {
            fn $fn_(&mut self, rhs: &T) {
                $( std::ops::$trait_::$fn_(&mut self.$field, *rhs); )*
            }
        }
    };
}

impl_op_assign!(Vector { x, y, z });
impl_op_assign!(Bivector { xy, yz, xz });
impl_op_assign!(Trivector { xyz });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_axis_angle() {
        let axis = Vector::new(1., 0., 0.);
        let angle = std::f64::consts::FRAC_PI_4;

        let motor = Motor::from_axis_angle(axis, angle);
        let (ang, ax) = motor.angle_axis();

        assert!((angle - ang).abs() < 0.00000001);
        assert!(axis.dot(ax) > 0.99999999);
    }

    #[test]
    fn rotor_sqr_one() {
        let one = Multivector(1., Zero, Bivector::new(0., 0., 0.), Zero);
        assert_eq!(one, one.sqrt());
    }

    #[test]
    fn rotor_sqr_quarter_turn() {
        let xy = Vector::new(1., 0., 0.).wedge(Vector::new(0., 1., 0.));
        let quarter = Multivector(0., Zero, xy, Zero);
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let eighth = Multivector(r, Zero, xy * r, Zero);
        assert_eq!(eighth, quarter.sqrt());
    }

    #[test]
    fn rotor_axis_angle() {
        let xy = Vector::new(1., 0., 0.).wedge(Vector::new(0., 1., 0.));
        let half = Multivector(0., Zero, xy, Zero);
        let quarter: Motor<f64> = half.sqrt();
        let eighth = quarter.sqrt();

        let (angle, axis) = quarter.angle_axis();
        assert_eq!(std::f64::consts::FRAC_PI_2, angle);
        assert_eq!(xy.dual(), axis);

        let (angle, axis) = eighth.angle_axis();
        assert!(std::f64::consts::FRAC_PI_4.sub(angle).abs() < 0.0000000001);
        assert_eq!(xy.dual(), axis);

        let (angle, axis) = quarter.rev().angle_axis();
        assert_eq!(std::f64::consts::FRAC_PI_2, angle);
        assert_eq!(-xy.dual(), axis);

        let neg_eighth: Motor<f64> = half.rev().sqrt();
        let (angle, axis) = neg_eighth.angle_axis();
        assert_eq!(std::f64::consts::FRAC_PI_2, angle);
        assert_eq!(-xy.dual(), axis);

        assert_eq!(
            Vector::new(-1., 0., 0.),
            (xy.geo(Vector::new(1., 0., 0.)).geo(xy.rev())).1
        );
    }

    #[test]
    fn vector_autodif_test() {
        let x = Vector::new(1f32, 2., 3.);
        let v = Vector::new(-5f32, 7., 11.);

        let ux = x / x.dot(x).sqrt();
        let uv = v / v.dot(v).sqrt();

        let p = ux.geo(uv);
        dbg!(p);

        // panic!("{:?}", p);
    }

    #[test]
    fn accel_derivative() {
        let a = |x: Vector<f64>| -x / x.dot(x).powf(1.5);
        let x_0 = Vector::new(0f64, 0., 2.);

        let norm = |v: Vector<f64>| v.dot(v).sqrt();

        let da = |x: Vector<f64>| {
            let inv_det = x.dot(x).powf(-2.5);
            let Vector { x, y, z } = x;
            let (x2, y2, z2) = (x * x, y * y, z * z);
            let da = Vector::new(
                -2. * x2 + y2 + z2 - 3. * x * (y + z),
                x2 - 2. * y2 + z2 - 3. * y * (x + z),
                x2 + y2 - 2. * z2 - 3. * z * (x + y),
            ) * inv_det;
            move |h: Vector<f64>| Vector::new(da.x * h.x, da.y * h.y, da.z * h.z) / norm(h)
        };

        let dt = 0.05f64;
        let dt2 = dt * dt;
        let mut x = x_0;
        let mut v = Vector::new(0f64, 0.01, 0.02);

        dbg!(x, v);
        for _ in 0..20 {
            let a = a(x) + 0.5 * da(x)(v * dt);

            x = x + dt * v + 0.5 * dt2 * a;
            v = v + dt * a;

            dbg!(x, v);
        }

        // panic!("done");
    }

    #[test]
    fn g3_contains_dual() {
        let vector = Vector::new(1., 2., 3.);
        let expected = Bivector::new(3., -2., 1.);
        assert_eq!(expected, vector.dual());
    }

    #[test]
    fn dual_is_symmetrical() {
        let vector = Vector::new(1., 2., 3.);
        assert_eq!(vector, vector.dual().dual());
    }
}

pub trait Geometric<Rhs> {
    type Output;
    fn geo(self, rhs: Rhs) -> Self::Output;
}
pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}
pub trait Wedge<Rhs> {
    type Output;
    fn wedge(self, rhs: Rhs) -> Self::Output;
}
pub trait Antigeometric<Rhs> {
    type Output;
    fn antigeo(self, rhs: Rhs) -> Self::Output;
}
pub trait Antidot<Rhs> {
    type Output;
    fn antidot(self, rhs: Rhs) -> Self::Output;
}
pub trait Antiwedge<Rhs> {
    type Output;
    fn antiwedge(self, rhs: Rhs) -> Self::Output;
}
pub trait LeftContraction<Rhs> {
    type Output;
    fn left_contraction(self, rhs: Rhs) -> Self::Output;
}
pub trait RightContraction<Rhs> {
    type Output;
    fn right_contraction(self, rhs: Rhs) -> Self::Output;
}
pub trait ScalarProduct<Rhs> {
    type Output;
    fn scalar_prod(self, rhs: Rhs) -> Self::Output;
}
pub trait VectorProduct<Rhs> {
    type Output;
    fn vector_prod(self, rhs: Rhs) -> Self::Output;
}
pub trait BivectorProduct<Rhs> {
    type Output;
    fn bivector_prod(self, rhs: Rhs) -> Self::Output;
}
pub trait TrivectorProduct<Rhs> {
    type Output;
    fn trivector_prod(self, rhs: Rhs) -> Self::Output;
}
impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antigeometric<Rhs> for Lhs
where
    Lhs: Dual<Output = LhsComp>,
    Rhs: Dual<Output = RhsComp>,
    LhsComp: Geometric<RhsComp, Output = OutputComp>,
    OutputComp: Dual,
{
    type Output = OutputComp::Output;
    #[inline]
    fn antigeo(self, rhs: Rhs) -> Self::Output {
        let lhs = self.dual();
        let rhs = rhs.dual();
        let output_complement = lhs.geo(rhs);
        output_complement.dual()
    }
}
impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antidot<Rhs> for Lhs
where
    Lhs: Dual<Output = LhsComp>,
    Rhs: Dual<Output = RhsComp>,
    LhsComp: Dot<RhsComp, Output = OutputComp>,
    OutputComp: Dual,
{
    type Output = OutputComp::Output;
    #[inline]
    fn antidot(self, rhs: Rhs) -> Self::Output {
        let lhs = self.dual();
        let rhs = rhs.dual();
        let output_complement = lhs.dot(rhs);
        output_complement.dual()
    }
}
impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> Antiwedge<Rhs> for Lhs
where
    Lhs: Dual<Output = LhsComp>,
    Rhs: Dual<Output = RhsComp>,
    LhsComp: Wedge<RhsComp, Output = OutputComp>,
    OutputComp: Dual,
{
    type Output = OutputComp::Output;
    #[inline]
    fn antiwedge(self, rhs: Rhs) -> Self::Output {
        let lhs = self.dual();
        let rhs = rhs.dual();
        let output_complement = lhs.wedge(rhs);
        output_complement.dual()
    }
}
pub trait Reverse {
    type Output;
    fn rev(self) -> Self::Output;
}
pub trait Antireverse {
    type Output;
    fn antirev(self) -> Self::Output;
}
pub trait Dual {
    type Output;
    fn dual(self) -> Self::Output;
}
impl<T, U> Antireverse for T
where
    T: Dual<Output = U>,
    U: Dual<Output = T> + Reverse<Output = U>,
{
    type Output = Self;
    fn antirev(self) -> Self::Output {
        self.dual().rev().dual()
    }
}
pub trait GradeAdd<Rhs> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
pub trait GradeSub<Rhs> {
    type Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
}
pub trait ToF32 {
    type Output;
    fn to_f32(self) -> Self::Output;
}
pub trait ToF64 {
    type Output;
    fn to_f64(self) -> Self::Output;
}
pub trait Norm {
    type Output;
    fn norm(self) -> Self::Output;
}

impl<T: Float> Norm for Vector<T> {
    type Output = T;
    #[inline]
    fn norm(self) -> Self::Output {
        self.norm2().sqrt()
    }
}

impl<T: Float> Norm for Bivector<T>
where
    Self: Norm2<Output = T>,
{
    type Output = T;
    #[inline]
    fn norm(self) -> Self::Output {
        self.norm2().sqrt()
    }
}

pub trait Norm2 {
    type Output;
    fn norm2(self) -> Self::Output;
}
impl<T, S> Norm2 for T
where
    T: ScalarProduct<T, Output = S> + Reverse<Output = T> + Copy,
{
    type Output = S;
    #[inline]
    fn norm2(self) -> Self::Output {
        self.scalar_prod(self.rev())
    }
}
pub trait Inverse {
    fn inv(self) -> Self;
}
impl<T, S> Inverse for T
where
    T: Reverse<Output = T> + Norm2<Output = S> + std::ops::Div<S, Output = T> + Copy,
    S: num_traits::Float,
{
    #[inline]
    fn inv(self) -> Self {
        let norm2 = self.norm2();
        if norm2 == S::zero() {
            panic!("div by zero");
        }
        self.rev() / norm2
    }
}
pub trait Unit {
    type Output;
    fn unit(self) -> Self::Output;
}

impl<T: Float> Unit for Vector<T> {
    type Output = Self;

    fn unit(self) -> Self::Output {
        self / self.norm()
    }
}

impl<T: Float> Unit for Bivector<T> {
    type Output = Self;

    fn unit(self) -> Self::Output {
        self / self.norm()
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Zero;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vector<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Bivector<T> {
    pub xy: T,
    pub xz: T,
    pub yz: T,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Trivector<T> {
    pub xyz: T,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Multivector<G0, G1, G2, G3>(pub G0, pub G1, pub G2, pub G3);
impl<T> Vector<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Vector<T> {
        Vector { x, y, z }
    }
}
impl<T> Bivector<T> {
    #[inline]
    pub const fn new(xy: T, xz: T, yz: T) -> Bivector<T> {
        Bivector { xy, xz, yz }
    }
}
impl<T> Trivector<T> {
    #[inline]
    pub const fn new(xyz: T) -> Trivector<T> {
        Trivector { xyz }
    }
}
impl std::ops::Add<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl std::ops::Add<f32> for Zero {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: f32) -> Self::Output {
        rhs
    }
}
impl std::ops::Add<f64> for Zero {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: f64) -> Self::Output {
        rhs
    }
}
impl<T> std::ops::Add<Vector<T>> for Zero {
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        rhs
    }
}
impl<T> std::ops::Add<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        rhs
    }
}
impl<T> std::ops::Add<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        rhs
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero {
    type Output = Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        rhs
    }
}
impl std::ops::Add<Zero> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl std::ops::Add<Zero> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl std::ops::Add<Vector<f32>> for f32 {
    type Output = Multivector<f32, Vector<f32>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<f32>) -> Self::Output {
        Multivector(self, rhs, Zero, Zero)
    }
}
impl std::ops::Add<Vector<f64>> for f64 {
    type Output = Multivector<f64, Vector<f64>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<f64>) -> Self::Output {
        Multivector(self, rhs, Zero, Zero)
    }
}
impl std::ops::Add<Bivector<f32>> for f32 {
    type Output = Multivector<f32, Zero, Bivector<f32>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<f32>) -> Self::Output {
        Multivector(self, Zero, rhs, Zero)
    }
}
impl std::ops::Add<Bivector<f64>> for f64 {
    type Output = Multivector<f64, Zero, Bivector<f64>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<f64>) -> Self::Output {
        Multivector(self, Zero, rhs, Zero)
    }
}
impl std::ops::Add<Trivector<f32>> for f32 {
    type Output = Multivector<f32, Zero, Zero, Trivector<f32>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<f32>) -> Self::Output {
        Multivector(self, Zero, Zero, rhs)
    }
}
impl std::ops::Add<Trivector<f64>> for f64 {
    type Output = Multivector<f64, Zero, Zero, Trivector<f64>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<f64>) -> Self::Output {
        Multivector(self, Zero, Zero, rhs)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: GradeAdd<G0Rhs, Output = G0Out>,
{
    type Output = Multivector<G0Out, G1Rhs, G2Rhs, G3Rhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(GradeAdd::add(self, rhs.0), rhs.1, rhs.2, rhs.3)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: GradeAdd<G0Rhs, Output = G0Out>,
{
    type Output = Multivector<G0Out, G1Rhs, G2Rhs, G3Rhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(GradeAdd::add(self, rhs.0), rhs.1, rhs.2, rhs.3)
    }
}
impl<T> std::ops::Add<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Add<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, self, Zero, Zero)
    }
}
impl<T> std::ops::Add<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(self.x, rhs.x),
                y: std::intrinsics::fadd_fast(self.y, rhs.y),
                z: std::intrinsics::fadd_fast(self.z, rhs.z),
            }
        }
    }
}
impl<T> std::ops::Add<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, self, rhs, Zero)
    }
}
impl<T> std::ops::Add<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, self, Zero, rhs)
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: GradeAdd<G1Rhs, Output = G1Out>,
{
    type Output = Multivector<G0Rhs, G1Out, G2Rhs, G3Rhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(rhs.0, GradeAdd::add(self, rhs.1), rhs.2, rhs.3)
    }
}
impl<T> std::ops::Add<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Add<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, Zero, self, Zero)
    }
}
impl<T> std::ops::Add<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, rhs, self, Zero)
    }
}
impl<T> std::ops::Add<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fadd_fast(self.xy, rhs.xy),
                xz: std::intrinsics::fadd_fast(self.xz, rhs.xz),
                yz: std::intrinsics::fadd_fast(self.yz, rhs.yz),
            }
        }
    }
}
impl<T> std::ops::Add<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, Zero, self, rhs)
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: GradeAdd<G2Rhs, Output = G2Out>,
{
    type Output = Multivector<G0Rhs, G1Rhs, G2Out, G3Rhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(rhs.0, rhs.1, GradeAdd::add(self, rhs.2), rhs.3)
    }
}
impl<T> std::ops::Add<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Add<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, Zero, Zero, self)
    }
}
impl<T> std::ops::Add<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, rhs, Zero, self)
    }
}
impl<T> std::ops::Add<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, Zero, rhs, self)
    }
}
impl<T> std::ops::Add<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(self.xyz, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: GradeAdd<G3Rhs, Output = G3Out>,
{
    type Output = Multivector<G0Rhs, G1Rhs, G2Rhs, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(rhs.0, rhs.1, rhs.2, GradeAdd::add(self, rhs.3))
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> std::ops::Add<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs> {
    type Output = Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0Out> std::ops::Add<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: GradeAdd<T, Output = G0Out>,
{
    type Output = Multivector<G0Out, G1Lhs, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(GradeAdd::add(self.0, rhs), self.1, self.2, self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1Out> std::ops::Add<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: GradeAdd<Vector<T>, Output = G1Out>,
{
    type Output = Multivector<G0Lhs, G1Out, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(self.0, GradeAdd::add(self.1, rhs), self.2, self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2Out> std::ops::Add<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: GradeAdd<Bivector<T>, Output = G2Out>,
{
    type Output = Multivector<G0Lhs, G1Lhs, G2Out, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(self.0, self.1, GradeAdd::add(self.2, rhs), self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3Out> std::ops::Add<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: GradeAdd<Trivector<T>, Output = G3Out>,
{
    type Output = Multivector<G0Lhs, G1Lhs, G2Lhs, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(self.0, self.1, self.2, GradeAdd::add(self.3, rhs))
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: GradeAdd<G0Rhs, Output = G0Out>,
    G1Lhs: GradeAdd<G1Rhs, Output = G1Out>,
    G2Lhs: GradeAdd<G2Rhs, Output = G2Out>,
    G3Lhs: GradeAdd<G3Rhs, Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            GradeAdd::add(self.0, rhs.0),
            GradeAdd::add(self.1, rhs.1),
            GradeAdd::add(self.2, rhs.2),
            GradeAdd::add(self.3, rhs.3),
        )
    }
}
impl std::ops::Sub<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl std::ops::Sub<f32> for Zero {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: f32) -> Self::Output {
        -rhs
    }
}
impl std::ops::Sub<f64> for Zero {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: f64) -> Self::Output {
        -rhs
    }
}
impl<T> std::ops::Sub<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        -rhs
    }
}
impl<T> std::ops::Sub<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        -rhs
    }
}
impl<T> std::ops::Sub<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        -rhs
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: std::ops::Neg<Output = G0Out>,
    G1Rhs: std::ops::Neg<Output = G1Out>,
    G2Rhs: std::ops::Neg<Output = G2Out>,
    G3Rhs: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        -rhs
    }
}
impl std::ops::Sub<Zero> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl std::ops::Sub<Zero> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl std::ops::Sub<Vector<f32>> for f32 {
    type Output = Multivector<f32, Vector<f32>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<f32>) -> Self::Output {
        Multivector(self, -rhs, Zero, Zero)
    }
}
impl std::ops::Sub<Vector<f64>> for f64 {
    type Output = Multivector<f64, Vector<f64>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<f64>) -> Self::Output {
        Multivector(self, -rhs, Zero, Zero)
    }
}
impl std::ops::Sub<Bivector<f32>> for f32 {
    type Output = Multivector<f32, Zero, Bivector<f32>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<f32>) -> Self::Output {
        Multivector(self, Zero, -rhs, Zero)
    }
}
impl std::ops::Sub<Bivector<f64>> for f64 {
    type Output = Multivector<f64, Zero, Bivector<f64>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<f64>) -> Self::Output {
        Multivector(self, Zero, -rhs, Zero)
    }
}
impl std::ops::Sub<Trivector<f32>> for f32 {
    type Output = Multivector<f32, Zero, Zero, Trivector<f32>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<f32>) -> Self::Output {
        Multivector(self, Zero, Zero, -rhs)
    }
}
impl std::ops::Sub<Trivector<f64>> for f64 {
    type Output = Multivector<f64, Zero, Zero, Trivector<f64>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<f64>) -> Self::Output {
        Multivector(self, Zero, Zero, -rhs)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: GradeSub<G0Rhs, Output = G0Out>,
    G1Rhs: std::ops::Neg<Output = G1Out>,
    G2Rhs: std::ops::Neg<Output = G2Out>,
    G3Rhs: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(GradeSub::sub(self, rhs.0), -rhs.1, -rhs.2, -rhs.3)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: GradeSub<G0Rhs, Output = G0Out>,
    G1Rhs: std::ops::Neg<Output = G1Out>,
    G2Rhs: std::ops::Neg<Output = G2Out>,
    G3Rhs: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(GradeSub::sub(self, rhs.0), -rhs.1, -rhs.2, -rhs.3)
    }
}
impl<T> std::ops::Sub<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Sub<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, self, Zero, Zero)
    }
}
impl<T> std::ops::Sub<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fsub_fast(self.x, rhs.x),
                y: std::intrinsics::fsub_fast(self.y, rhs.y),
                z: std::intrinsics::fsub_fast(self.z, rhs.z),
            }
        }
    }
}
impl<T> std::ops::Sub<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, self, -rhs, Zero)
    }
}
impl<T> std::ops::Sub<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, self, Zero, -rhs)
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    G0Rhs: std::ops::Neg<Output = G0Out>,
    Vector<T>: GradeSub<G1Rhs, Output = G1Out>,
    G2Rhs: std::ops::Neg<Output = G2Out>,
    G3Rhs: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(-rhs.0, GradeSub::sub(self, rhs.1), -rhs.2, -rhs.3)
    }
}
impl<T> std::ops::Sub<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Sub<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, Zero, self, Zero)
    }
}
impl<T> std::ops::Sub<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, -rhs, self, Zero)
    }
}
impl<T> std::ops::Sub<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fsub_fast(self.xy, rhs.xy),
                xz: std::intrinsics::fsub_fast(self.xz, rhs.xz),
                yz: std::intrinsics::fsub_fast(self.yz, rhs.yz),
            }
        }
    }
}
impl<T> std::ops::Sub<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, Zero, self, -rhs)
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    G0Rhs: std::ops::Neg<Output = G0Out>,
    G1Rhs: std::ops::Neg<Output = G1Out>,
    Bivector<T>: GradeSub<G2Rhs, Output = G2Out>,
    G3Rhs: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(-rhs.0, -rhs.1, GradeSub::sub(self, rhs.2), -rhs.3)
    }
}
impl<T> std::ops::Sub<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> std::ops::Sub<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, Zero, Zero, self)
    }
}
impl<T> std::ops::Sub<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, -rhs, Zero, self)
    }
}
impl<T> std::ops::Sub<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, Zero, -rhs, self)
    }
}
impl<T> std::ops::Sub<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fsub_fast(self.xyz, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Trivector<T>
where
    T: num_traits::Float,
    G0Rhs: std::ops::Neg<Output = G0Out>,
    G1Rhs: std::ops::Neg<Output = G1Out>,
    G2Rhs: std::ops::Neg<Output = G2Out>,
    Trivector<T>: GradeSub<G3Rhs, Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(-rhs.0, -rhs.1, -rhs.2, GradeSub::sub(self, rhs.3))
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> std::ops::Sub<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs> {
    type Output = Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0Out> std::ops::Sub<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: GradeSub<T, Output = G0Out>,
{
    type Output = Multivector<G0Out, G1Lhs, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(GradeSub::sub(self.0, rhs), self.1, self.2, self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1Out> std::ops::Sub<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: GradeSub<Vector<T>, Output = G1Out>,
{
    type Output = Multivector<G0Lhs, G1Out, G2Lhs, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(self.0, GradeSub::sub(self.1, rhs), self.2, self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2Out> std::ops::Sub<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: GradeSub<Bivector<T>, Output = G2Out>,
{
    type Output = Multivector<G0Lhs, G1Lhs, G2Out, G3Lhs>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(self.0, self.1, GradeSub::sub(self.2, rhs), self.3)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3Out> std::ops::Sub<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: GradeSub<Trivector<T>, Output = G3Out>,
{
    type Output = Multivector<G0Lhs, G1Lhs, G2Lhs, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(self.0, self.1, self.2, GradeSub::sub(self.3, rhs))
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0Out, G1Out, G2Out, G3Out>
    std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: GradeSub<G0Rhs, Output = G0Out>,
    G1Lhs: GradeSub<G1Rhs, Output = G1Out>,
    G2Lhs: GradeSub<G2Rhs, Output = G2Out>,
    G3Lhs: GradeSub<G3Rhs, Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            GradeSub::sub(self.0, rhs.0),
            GradeSub::sub(self.1, rhs.1),
            GradeSub::sub(self.2, rhs.2),
            GradeSub::sub(self.3, rhs.3),
        )
    }
}
impl GradeAdd<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl GradeAdd<f32> for Zero {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: f32) -> Self::Output {
        rhs
    }
}
impl GradeAdd<f64> for Zero {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: f64) -> Self::Output {
        rhs
    }
}
impl<T> GradeAdd<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        rhs
    }
}
impl<T> GradeAdd<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        rhs
    }
}
impl<T> GradeAdd<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        rhs
    }
}
impl<T> GradeAdd<Zero> for T
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeAdd<T> for T
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        unsafe { std::intrinsics::fadd_fast(self, rhs) }
    }
}
impl<T> GradeAdd<Vector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(self, rhs, Zero, Zero)
    }
}
impl<T> GradeAdd<Bivector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(self, Zero, rhs, Zero)
    }
}
impl<T> GradeAdd<Trivector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(self, Zero, Zero, rhs)
    }
}
impl<T> GradeAdd<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeAdd<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, self, Zero, Zero)
    }
}
impl<T> GradeAdd<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(self.x, rhs.x),
                y: std::intrinsics::fadd_fast(self.y, rhs.y),
                z: std::intrinsics::fadd_fast(self.z, rhs.z),
            }
        }
    }
}
impl<T> GradeAdd<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, self, rhs, Zero)
    }
}
impl<T> GradeAdd<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, self, Zero, rhs)
    }
}
impl<T> GradeAdd<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeAdd<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, Zero, self, Zero)
    }
}
impl<T> GradeAdd<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, rhs, self, Zero)
    }
}
impl<T> GradeAdd<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fadd_fast(self.xy, rhs.xy),
                xz: std::intrinsics::fadd_fast(self.xz, rhs.xz),
                yz: std::intrinsics::fadd_fast(self.yz, rhs.yz),
            }
        }
    }
}
impl<T> GradeAdd<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, Zero, self, rhs)
    }
}
impl<T> GradeAdd<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeAdd<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: T) -> Self::Output {
        Multivector(rhs, Zero, Zero, self)
    }
}
impl<T> GradeAdd<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, rhs, Zero, self)
    }
}
impl<T> GradeAdd<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, Zero, rhs, self)
    }
}
impl<T> GradeAdd<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn add(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(self.xyz, rhs.xyz),
            }
        }
    }
}
impl GradeSub<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl GradeSub<f32> for Zero {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: f32) -> Self::Output {
        -rhs
    }
}
impl GradeSub<f64> for Zero {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: f64) -> Self::Output {
        -rhs
    }
}
impl<T> GradeSub<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        -rhs
    }
}
impl<T> GradeSub<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        -rhs
    }
}
impl<T> GradeSub<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        -rhs
    }
}
impl<T> GradeSub<Zero> for T
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeSub<T> for T
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        unsafe { std::intrinsics::fsub_fast(self, rhs) }
    }
}
impl<T> GradeSub<Vector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(self, -rhs, Zero, Zero)
    }
}
impl<T> GradeSub<Bivector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(self, Zero, -rhs, Zero)
    }
}
impl<T> GradeSub<Trivector<T>> for T
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(self, Zero, Zero, -rhs)
    }
}
impl<T> GradeSub<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeSub<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Vector<T>, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, self, Zero, Zero)
    }
}
impl<T> GradeSub<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fsub_fast(self.x, rhs.x),
                y: std::intrinsics::fsub_fast(self.y, rhs.y),
                z: std::intrinsics::fsub_fast(self.z, rhs.z),
            }
        }
    }
}
impl<T> GradeSub<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, self, -rhs, Zero)
    }
}
impl<T> GradeSub<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, self, Zero, -rhs)
    }
}
impl<T> GradeSub<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeSub<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, Zero, self, Zero)
    }
}
impl<T> GradeSub<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, -rhs, self, Zero)
    }
}
impl<T> GradeSub<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fsub_fast(self.xy, rhs.xy),
                xz: std::intrinsics::fsub_fast(self.xz, rhs.xz),
                yz: std::intrinsics::fsub_fast(self.yz, rhs.yz),
            }
        }
    }
}
impl<T> GradeSub<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(Zero, Zero, self, -rhs)
    }
}
impl<T> GradeSub<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Zero) -> Self::Output {
        self
    }
}
impl<T> GradeSub<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: T) -> Self::Output {
        Multivector(-rhs, Zero, Zero, self)
    }
}
impl<T> GradeSub<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, -rhs, Zero, self)
    }
}
impl<T> GradeSub<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Zero, Bivector<T>, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, Zero, -rhs, self)
    }
}
impl<T> GradeSub<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn sub(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fsub_fast(self.xyz, rhs.xyz),
            }
        }
    }
}
impl std::ops::Mul<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl std::ops::Mul<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl std::ops::Mul<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl std::ops::Mul<Vector<f32>> for f32 {
    type Output = Vector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Vector<f32>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl std::ops::Mul<Vector<f64>> for f64 {
    type Output = Vector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Vector<f64>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl std::ops::Mul<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl std::ops::Mul<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl std::ops::Mul<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl std::ops::Mul<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: ScalarProduct<G0Rhs, Output = G0_0>,
    f32: VectorProduct<G1Rhs, Output = G1_0>,
    f32: BivectorProduct<G2Rhs, Output = G2_0>,
    f32: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: ScalarProduct<G0Rhs, Output = G0_0>,
    f64: VectorProduct<G1Rhs, Output = G1_0>,
    f64: BivectorProduct<G2Rhs, Output = G2_0>,
    f64: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<T> std::ops::Mul<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
// impl<T> std::ops::Mul<Vector<T>> for Vector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Multivector<T, Zero, Bivector<T>, Zero>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Vector<T>) -> Self::Output {
//         Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
//     }
// }
// impl<T> std::ops::Mul<Bivector<T>> for Vector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Bivector<T>) -> Self::Output {
//         Multivector(Zero, self.vector_prod(rhs), Zero, self.trivector_prod(rhs))
//     }
// }
// impl<T> std::ops::Mul<Trivector<T>> for Vector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Bivector<T>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Trivector<T>) -> Self::Output {
//         unsafe {
//             Bivector {
//                 xy: std::intrinsics::fmul_fast(self.z, rhs.xyz),
//                 xz: -std::intrinsics::fmul_fast(self.y, rhs.xyz),
//                 yz: std::intrinsics::fmul_fast(self.x, rhs.xyz),
//             }
//         }
//     }
// }
// impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
//     std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
// where
//     T: num_traits::Float,
//     Vector<T>: ScalarProduct<G1Rhs, Output = G0_0>,
//     Vector<T>: VectorProduct<G0Rhs, Output = G1_0>,
//     Vector<T>: VectorProduct<G2Rhs, Output = G1_1>,
//     Vector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
//     Vector<T>: BivectorProduct<G3Rhs, Output = G2_1>,
//     Vector<T>: TrivectorProduct<G2Rhs, Output = G3_0>,
//     G0Rhs: Copy,
//     G1Rhs: Copy,
//     G2Rhs: Copy,
//     G3Rhs: Copy,
//     G1_0: std::ops::Add<G1_1, Output = G1_2>,
//     G2_0: std::ops::Add<G2_1, Output = G2_2>,
// {
//     type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
//         Multivector(
//             self.scalar_prod(rhs.1),
//             self.vector_prod(rhs.0) + self.vector_prod(rhs.2),
//             self.bivector_prod(rhs.1) + self.bivector_prod(rhs.3),
//             self.trivector_prod(rhs.2),
//         )
//     }
// }
impl<T> std::ops::Mul<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
// impl<T> std::ops::Mul<Vector<T>> for Bivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Vector<T>) -> Self::Output {
//         Multivector(Zero, self.vector_prod(rhs), Zero, self.trivector_prod(rhs))
//     }
// }
// impl<T> std::ops::Mul<Bivector<T>> for Bivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Multivector<T, Zero, Bivector<T>, Zero>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Bivector<T>) -> Self::Output {
//         Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
//     }
// }
// impl<T> std::ops::Mul<Trivector<T>> for Bivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Vector<T>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Trivector<T>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: -std::intrinsics::fmul_fast(self.yz, rhs.xyz),
//                 y: std::intrinsics::fmul_fast(self.xz, rhs.xyz),
//                 z: -std::intrinsics::fmul_fast(self.xy, rhs.xyz),
//             }
//         }
//     }
// }
// impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
//     std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
// where
//     T: num_traits::Float,
//     Bivector<T>: ScalarProduct<G2Rhs, Output = G0_0>,
//     Bivector<T>: VectorProduct<G1Rhs, Output = G1_0>,
//     Bivector<T>: VectorProduct<G3Rhs, Output = G1_1>,
//     Bivector<T>: BivectorProduct<G0Rhs, Output = G2_0>,
//     Bivector<T>: BivectorProduct<G2Rhs, Output = G2_1>,
//     Bivector<T>: TrivectorProduct<G1Rhs, Output = G3_0>,
//     G0Rhs: Copy,
//     G1Rhs: Copy,
//     G2Rhs: Copy,
//     G3Rhs: Copy,
//     G1_0: std::ops::Add<G1_1, Output = G1_2>,
//     G2_0: std::ops::Add<G2_1, Output = G2_2>,
// {
//     type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
//         Multivector(
//             self.scalar_prod(rhs.2),
//             self.vector_prod(rhs.1) + self.vector_prod(rhs.3),
//             self.bivector_prod(rhs.0) + self.bivector_prod(rhs.2),
//             self.trivector_prod(rhs.1),
//         )
//     }
// }
impl<T> std::ops::Mul<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Mul<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
// impl<T> std::ops::Mul<Vector<T>> for Trivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Bivector<T>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Vector<T>) -> Self::Output {
//         unsafe {
//             Bivector {
//                 xy: std::intrinsics::fmul_fast(self.xyz, rhs.z),
//                 xz: -std::intrinsics::fmul_fast(self.xyz, rhs.y),
//                 yz: std::intrinsics::fmul_fast(self.xyz, rhs.x),
//             }
//         }
//     }
// }
// impl<T> std::ops::Mul<Bivector<T>> for Trivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Vector<T>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Bivector<T>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: -std::intrinsics::fmul_fast(self.xyz, rhs.yz),
//                 y: std::intrinsics::fmul_fast(self.xyz, rhs.xz),
//                 z: -std::intrinsics::fmul_fast(self.xyz, rhs.xy),
//             }
//         }
//     }
// }
impl<T> std::ops::Mul<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
// impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
//     std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Trivector<T>
// where
//     T: num_traits::Float,
//     Trivector<T>: ScalarProduct<G3Rhs, Output = G0_0>,
//     Trivector<T>: VectorProduct<G2Rhs, Output = G1_0>,
//     Trivector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
//     Trivector<T>: TrivectorProduct<G0Rhs, Output = G3_0>,
//     G0Rhs: Copy,
//     G1Rhs: Copy,
//     G2Rhs: Copy,
//     G3Rhs: Copy,
// {
//     type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
//         Multivector(
//             self.scalar_prod(rhs.3),
//             self.vector_prod(rhs.2),
//             self.bivector_prod(rhs.1),
//             self.trivector_prod(rhs.0),
//         )
//     }
// }
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> std::ops::Mul<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
// impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> std::ops::Mul<T>
//     for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
// where
//     T: num_traits::Float,
//     G0Lhs: ScalarProduct<T, Output = G0_0>,
//     G1Lhs: VectorProduct<T, Output = G1_0>,
//     G2Lhs: BivectorProduct<T, Output = G2_0>,
//     G3Lhs: TrivectorProduct<T, Output = G3_0>,
//     G0Lhs: Copy,
//     G1Lhs: Copy,
//     G2Lhs: Copy,
//     G3Lhs: Copy,
// {
//     type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: T) -> Self::Output {
//         Multivector(
//             self.0.scalar_prod(rhs),
//             self.1.vector_prod(rhs),
//             self.2.bivector_prod(rhs),
//             self.3.trivector_prod(rhs),
//         )
//     }
// }
// impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
//     std::ops::Mul<Vector<T>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
// where
//     T: num_traits::Float,
//     G1Lhs: ScalarProduct<Vector<T>, Output = G0_0>,
//     G0Lhs: VectorProduct<Vector<T>, Output = G1_0>,
//     G2Lhs: VectorProduct<Vector<T>, Output = G1_1>,
//     G1Lhs: BivectorProduct<Vector<T>, Output = G2_0>,
//     G3Lhs: BivectorProduct<Vector<T>, Output = G2_1>,
//     G2Lhs: TrivectorProduct<Vector<T>, Output = G3_0>,
//     G0Lhs: Copy,
//     G1Lhs: Copy,
//     G2Lhs: Copy,
//     G3Lhs: Copy,
//     G1_0: std::ops::Add<G1_1, Output = G1_2>,
//     G2_0: std::ops::Add<G2_1, Output = G2_2>,
// {
//     type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Vector<T>) -> Self::Output {
//         Multivector(
//             self.1.scalar_prod(rhs),
//             self.0.vector_prod(rhs) + self.2.vector_prod(rhs),
//             self.1.bivector_prod(rhs) + self.3.bivector_prod(rhs),
//             self.2.trivector_prod(rhs),
//         )
//     }
// }
// impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
//     std::ops::Mul<Bivector<T>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
// where
//     T: num_traits::Float,
//     G2Lhs: ScalarProduct<Bivector<T>, Output = G0_0>,
//     G1Lhs: VectorProduct<Bivector<T>, Output = G1_0>,
//     G3Lhs: VectorProduct<Bivector<T>, Output = G1_1>,
//     G0Lhs: BivectorProduct<Bivector<T>, Output = G2_0>,
//     G2Lhs: BivectorProduct<Bivector<T>, Output = G2_1>,
//     G1Lhs: TrivectorProduct<Bivector<T>, Output = G3_0>,
//     G0Lhs: Copy,
//     G1Lhs: Copy,
//     G2Lhs: Copy,
//     G3Lhs: Copy,
//     G1_0: std::ops::Add<G1_1, Output = G1_2>,
//     G2_0: std::ops::Add<G2_1, Output = G2_2>,
// {
//     type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Bivector<T>) -> Self::Output {
//         Multivector(
//             self.2.scalar_prod(rhs),
//             self.1.vector_prod(rhs) + self.3.vector_prod(rhs),
//             self.0.bivector_prod(rhs) + self.2.bivector_prod(rhs),
//             self.1.trivector_prod(rhs),
//         )
//     }
// }
// impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> std::ops::Mul<Trivector<T>>
//     for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
// where
//     T: num_traits::Float,
//     G3Lhs: ScalarProduct<Trivector<T>, Output = G0_0>,
//     G2Lhs: VectorProduct<Trivector<T>, Output = G1_0>,
//     G1Lhs: BivectorProduct<Trivector<T>, Output = G2_0>,
//     G0Lhs: TrivectorProduct<Trivector<T>, Output = G3_0>,
//     G0Lhs: Copy,
//     G1Lhs: Copy,
//     G2Lhs: Copy,
//     G3Lhs: Copy,
// {
//     type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn mul(self, rhs: Trivector<T>) -> Self::Output {
//         Multivector(
//             self.3.scalar_prod(rhs),
//             self.2.vector_prod(rhs),
//             self.1.bivector_prod(rhs),
//             self.0.trivector_prod(rhs),
//         )
//     }
// }
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G1_5,
        G1_6,
        G1_7,
        G1_8,
        G1_9,
        G1_10,
        G2_0,
        G2_1,
        G2_2,
        G2_3,
        G2_4,
        G2_5,
        G2_6,
        G2_7,
        G2_8,
        G2_9,
        G2_10,
        G3_0,
        G3_1,
        G3_2,
        G3_3,
        G3_4,
        G3_5,
        G3_6,
    > std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ScalarProduct<G0Rhs, Output = G0_0>,
    G1Lhs: ScalarProduct<G1Rhs, Output = G0_1>,
    G2Lhs: ScalarProduct<G2Rhs, Output = G0_2>,
    G3Lhs: ScalarProduct<G3Rhs, Output = G0_3>,
    G0Lhs: VectorProduct<G1Rhs, Output = G1_0>,
    G1Lhs: VectorProduct<G0Rhs, Output = G1_1>,
    G1Lhs: VectorProduct<G2Rhs, Output = G1_2>,
    G2Lhs: VectorProduct<G1Rhs, Output = G1_3>,
    G2Lhs: VectorProduct<G3Rhs, Output = G1_4>,
    G3Lhs: VectorProduct<G2Rhs, Output = G1_5>,
    G0Lhs: BivectorProduct<G2Rhs, Output = G2_0>,
    G1Lhs: BivectorProduct<G1Rhs, Output = G2_1>,
    G1Lhs: BivectorProduct<G3Rhs, Output = G2_2>,
    G2Lhs: BivectorProduct<G0Rhs, Output = G2_3>,
    G2Lhs: BivectorProduct<G2Rhs, Output = G2_4>,
    G3Lhs: BivectorProduct<G1Rhs, Output = G2_5>,
    G0Lhs: TrivectorProduct<G3Rhs, Output = G3_0>,
    G1Lhs: TrivectorProduct<G2Rhs, Output = G3_1>,
    G2Lhs: TrivectorProduct<G1Rhs, Output = G3_2>,
    G3Lhs: TrivectorProduct<G0Rhs, Output = G3_3>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
    G1_0: std::ops::Add<G1_1, Output = G1_6>,
    G1_6: std::ops::Add<G1_2, Output = G1_7>,
    G1_7: std::ops::Add<G1_3, Output = G1_8>,
    G1_8: std::ops::Add<G1_4, Output = G1_9>,
    G1_9: std::ops::Add<G1_5, Output = G1_10>,
    G2_0: std::ops::Add<G2_1, Output = G2_6>,
    G2_6: std::ops::Add<G2_2, Output = G2_7>,
    G2_7: std::ops::Add<G2_3, Output = G2_8>,
    G2_8: std::ops::Add<G2_4, Output = G2_9>,
    G2_9: std::ops::Add<G2_5, Output = G2_10>,
    G3_0: std::ops::Add<G3_1, Output = G3_4>,
    G3_4: std::ops::Add<G3_2, Output = G3_5>,
    G3_5: std::ops::Add<G3_3, Output = G3_6>,
{
    type Output = Multivector<G0_6, G1_10, G2_10, G3_6>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs.0)
                + self.1.scalar_prod(rhs.1)
                + self.2.scalar_prod(rhs.2)
                + self.3.scalar_prod(rhs.3),
            self.0.vector_prod(rhs.1)
                + self.1.vector_prod(rhs.0)
                + self.1.vector_prod(rhs.2)
                + self.2.vector_prod(rhs.1)
                + self.2.vector_prod(rhs.3)
                + self.3.vector_prod(rhs.2),
            self.0.bivector_prod(rhs.2)
                + self.1.bivector_prod(rhs.1)
                + self.1.bivector_prod(rhs.3)
                + self.2.bivector_prod(rhs.0)
                + self.2.bivector_prod(rhs.2)
                + self.3.bivector_prod(rhs.1),
            self.0.trivector_prod(rhs.3)
                + self.1.trivector_prod(rhs.2)
                + self.2.trivector_prod(rhs.1)
                + self.3.trivector_prod(rhs.0),
        )
    }
}
// impl Geometric<Zero> for Zero {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl<T> Geometric<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> Geometric<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Geometric<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Geometric<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
// impl Geometric<Zero> for f32 {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
// impl Geometric<Zero> for f64 {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl Geometric<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl Geometric<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl Geometric<Vector<f32>> for f32 {
    type Output = Vector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<f32>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl Geometric<Vector<f64>> for f64 {
    type Output = Vector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<f64>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl Geometric<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Geometric<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Geometric<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl Geometric<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: ScalarProduct<G0Rhs, Output = G0_0>,
    f32: VectorProduct<G1Rhs, Output = G1_0>,
    f32: BivectorProduct<G2Rhs, Output = G2_0>,
    f32: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: ScalarProduct<G0Rhs, Output = G0_0>,
    f64: VectorProduct<G1Rhs, Output = G1_0>,
    f64: BivectorProduct<G2Rhs, Output = G2_0>,
    f64: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
// impl<T> Geometric<Zero> for Vector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl<T> Geometric<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
impl<T> Geometric<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<T>) -> Self::Output {
        Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
    }
}
impl<T> Geometric<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(Zero, self.vector_prod(rhs), Zero, self.trivector_prod(rhs))
    }
}
impl<T> Geometric<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.z, rhs.xyz),
                xz: -std::intrinsics::fmul_fast(self.y, rhs.xyz),
                yz: std::intrinsics::fmul_fast(self.x, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
    Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: ScalarProduct<G1Rhs, Output = G0_0>,
    Vector<T>: VectorProduct<G0Rhs, Output = G1_0>,
    Vector<T>: VectorProduct<G2Rhs, Output = G1_1>,
    Vector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    Vector<T>: BivectorProduct<G3Rhs, Output = G2_1>,
    Vector<T>: TrivectorProduct<G2Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.1),
            self.vector_prod(rhs.0) + self.vector_prod(rhs.2),
            self.bivector_prod(rhs.1) + self.bivector_prod(rhs.3),
            self.trivector_prod(rhs.2),
        )
    }
}
// impl<T> Geometric<Zero> for Bivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl<T> Geometric<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> Geometric<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<Zero, Vector<T>, Zero, Trivector<T>>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<T>) -> Self::Output {
        Multivector(Zero, self.vector_prod(rhs), Zero, self.trivector_prod(rhs))
    }
}
impl<T> Geometric<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Multivector<T, Zero, Bivector<T>, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
    }
}
impl<T> Geometric<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.yz, rhs.xyz),
                y: std::intrinsics::fmul_fast(self.xz, rhs.xyz),
                z: -std::intrinsics::fmul_fast(self.xy, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
    Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: ScalarProduct<G2Rhs, Output = G0_0>,
    Bivector<T>: VectorProduct<G1Rhs, Output = G1_0>,
    Bivector<T>: VectorProduct<G3Rhs, Output = G1_1>,
    Bivector<T>: BivectorProduct<G0Rhs, Output = G2_0>,
    Bivector<T>: BivectorProduct<G2Rhs, Output = G2_1>,
    Bivector<T>: TrivectorProduct<G1Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.2),
            self.vector_prod(rhs.1) + self.vector_prod(rhs.3),
            self.bivector_prod(rhs.0) + self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.1),
        )
    }
}
// impl<T> Geometric<Zero> for Trivector<T>
// where
//     T: num_traits::Float,
// {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl<T> Geometric<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T> Geometric<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xyz, rhs.z),
                xz: -std::intrinsics::fmul_fast(self.xyz, rhs.y),
                yz: std::intrinsics::fmul_fast(self.xyz, rhs.x),
            }
        }
    }
}
impl<T> Geometric<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.xyz, rhs.yz),
                y: std::intrinsics::fmul_fast(self.xyz, rhs.xz),
                z: -std::intrinsics::fmul_fast(self.xyz, rhs.xy),
            }
        }
    }
}
impl<T> Geometric<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: ScalarProduct<G3Rhs, Output = G0_0>,
    Trivector<T>: VectorProduct<G2Rhs, Output = G1_0>,
    Trivector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    Trivector<T>: TrivectorProduct<G0Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.3),
            self.vector_prod(rhs.2),
            self.bivector_prod(rhs.1),
            self.trivector_prod(rhs.0),
        )
    }
}
// impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> Geometric<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
// where
//     G0Lhs: Copy,
//     G1Lhs: Copy,
//     G2Lhs: Copy,
//     G3Lhs: Copy,
// {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn geo(self, rhs: Zero) -> Self::Output {
//         Zero
//     }
// }
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> Geometric<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: ScalarProduct<T, Output = G0_0>,
    G1Lhs: VectorProduct<T, Output = G1_0>,
    G2Lhs: BivectorProduct<T, Output = G2_0>,
    G3Lhs: TrivectorProduct<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: T) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs),
            self.1.vector_prod(rhs),
            self.2.bivector_prod(rhs),
            self.3.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
    Geometric<Vector<T>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: ScalarProduct<Vector<T>, Output = G0_0>,
    G0Lhs: VectorProduct<Vector<T>, Output = G1_0>,
    G2Lhs: VectorProduct<Vector<T>, Output = G1_1>,
    G1Lhs: BivectorProduct<Vector<T>, Output = G2_0>,
    G3Lhs: BivectorProduct<Vector<T>, Output = G2_1>,
    G2Lhs: TrivectorProduct<Vector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Vector<T>) -> Self::Output {
        Multivector(
            self.1.scalar_prod(rhs),
            self.0.vector_prod(rhs) + self.2.vector_prod(rhs),
            self.1.bivector_prod(rhs) + self.3.bivector_prod(rhs),
            self.2.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0, G2_1, G2_2, G3_0>
    Geometric<Bivector<T>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: ScalarProduct<Bivector<T>, Output = G0_0>,
    G1Lhs: VectorProduct<Bivector<T>, Output = G1_0>,
    G3Lhs: VectorProduct<Bivector<T>, Output = G1_1>,
    G0Lhs: BivectorProduct<Bivector<T>, Output = G2_0>,
    G2Lhs: BivectorProduct<Bivector<T>, Output = G2_1>,
    G1Lhs: TrivectorProduct<Bivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(
            self.2.scalar_prod(rhs),
            self.1.vector_prod(rhs) + self.3.vector_prod(rhs),
            self.0.bivector_prod(rhs) + self.2.bivector_prod(rhs),
            self.1.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> Geometric<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: ScalarProduct<Trivector<T>, Output = G0_0>,
    G2Lhs: VectorProduct<Trivector<T>, Output = G1_0>,
    G1Lhs: BivectorProduct<Trivector<T>, Output = G2_0>,
    G0Lhs: TrivectorProduct<Trivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(
            self.3.scalar_prod(rhs),
            self.2.vector_prod(rhs),
            self.1.bivector_prod(rhs),
            self.0.trivector_prod(rhs),
        )
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G1_5,
        G1_6,
        G1_7,
        G1_8,
        G1_9,
        G1_10,
        G2_0,
        G2_1,
        G2_2,
        G2_3,
        G2_4,
        G2_5,
        G2_6,
        G2_7,
        G2_8,
        G2_9,
        G2_10,
        G3_0,
        G3_1,
        G3_2,
        G3_3,
        G3_4,
        G3_5,
        G3_6,
    > Geometric<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ScalarProduct<G0Rhs, Output = G0_0>,
    G1Lhs: ScalarProduct<G1Rhs, Output = G0_1>,
    G2Lhs: ScalarProduct<G2Rhs, Output = G0_2>,
    G3Lhs: ScalarProduct<G3Rhs, Output = G0_3>,
    G0Lhs: VectorProduct<G1Rhs, Output = G1_0>,
    G1Lhs: VectorProduct<G0Rhs, Output = G1_1>,
    G1Lhs: VectorProduct<G2Rhs, Output = G1_2>,
    G2Lhs: VectorProduct<G1Rhs, Output = G1_3>,
    G2Lhs: VectorProduct<G3Rhs, Output = G1_4>,
    G3Lhs: VectorProduct<G2Rhs, Output = G1_5>,
    G0Lhs: BivectorProduct<G2Rhs, Output = G2_0>,
    G1Lhs: BivectorProduct<G1Rhs, Output = G2_1>,
    G1Lhs: BivectorProduct<G3Rhs, Output = G2_2>,
    G2Lhs: BivectorProduct<G0Rhs, Output = G2_3>,
    G2Lhs: BivectorProduct<G2Rhs, Output = G2_4>,
    G3Lhs: BivectorProduct<G1Rhs, Output = G2_5>,
    G0Lhs: TrivectorProduct<G3Rhs, Output = G3_0>,
    G1Lhs: TrivectorProduct<G2Rhs, Output = G3_1>,
    G2Lhs: TrivectorProduct<G1Rhs, Output = G3_2>,
    G3Lhs: TrivectorProduct<G0Rhs, Output = G3_3>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
    G1_0: std::ops::Add<G1_1, Output = G1_6>,
    G1_6: std::ops::Add<G1_2, Output = G1_7>,
    G1_7: std::ops::Add<G1_3, Output = G1_8>,
    G1_8: std::ops::Add<G1_4, Output = G1_9>,
    G1_9: std::ops::Add<G1_5, Output = G1_10>,
    G2_0: std::ops::Add<G2_1, Output = G2_6>,
    G2_6: std::ops::Add<G2_2, Output = G2_7>,
    G2_7: std::ops::Add<G2_3, Output = G2_8>,
    G2_8: std::ops::Add<G2_4, Output = G2_9>,
    G2_9: std::ops::Add<G2_5, Output = G2_10>,
    G3_0: std::ops::Add<G3_1, Output = G3_4>,
    G3_4: std::ops::Add<G3_2, Output = G3_5>,
    G3_5: std::ops::Add<G3_3, Output = G3_6>,
{
    type Output = Multivector<G0_6, G1_10, G2_10, G3_6>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn geo(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs.0)
                + self.1.scalar_prod(rhs.1)
                + self.2.scalar_prod(rhs.2)
                + self.3.scalar_prod(rhs.3),
            self.0.vector_prod(rhs.1)
                + self.1.vector_prod(rhs.0)
                + self.1.vector_prod(rhs.2)
                + self.2.vector_prod(rhs.1)
                + self.2.vector_prod(rhs.3)
                + self.3.vector_prod(rhs.2),
            self.0.bivector_prod(rhs.2)
                + self.1.bivector_prod(rhs.1)
                + self.1.bivector_prod(rhs.3)
                + self.2.bivector_prod(rhs.0)
                + self.2.bivector_prod(rhs.2)
                + self.3.bivector_prod(rhs.1),
            self.0.trivector_prod(rhs.3)
                + self.1.trivector_prod(rhs.2)
                + self.2.trivector_prod(rhs.1)
                + self.3.trivector_prod(rhs.0),
        )
    }
}
impl Dot<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Dot<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> Dot<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Dot<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Dot<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl Dot<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl Dot<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl Dot<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl Dot<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl<T: Float> Dot<Vector<T>> for T {
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: self * rhs.x,
                y: self * rhs.y,
                z: self * rhs.z,
            }
        }
    }
}
// impl Dot<Vector<f32>> for f32 {
//     type Output = Vector<f32>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn dot(self, rhs: Vector<f32>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: std::intrinsics::fmul_fast(self, rhs.x),
//                 y: std::intrinsics::fmul_fast(self, rhs.y),
//                 z: std::intrinsics::fmul_fast(self, rhs.z),
//             }
//         }
//     }
// }
// impl Dot<Vector<f64>> for f64 {
//     type Output = Vector<f64>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn dot(self, rhs: Vector<f64>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: std::intrinsics::fmul_fast(self, rhs.x),
//                 y: std::intrinsics::fmul_fast(self, rhs.y),
//                 z: std::intrinsics::fmul_fast(self, rhs.z),
//             }
//         }
//     }
// }
impl Dot<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Dot<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Dot<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl Dot<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: ScalarProduct<G0Rhs, Output = G0_0>,
    f32: VectorProduct<G1Rhs, Output = G1_0>,
    f32: BivectorProduct<G2Rhs, Output = G2_0>,
    f32: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: ScalarProduct<G0Rhs, Output = G0_0>,
    f64: VectorProduct<G1Rhs, Output = G1_0>,
    f64: BivectorProduct<G2Rhs, Output = G2_0>,
    f64: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<T> Dot<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Dot<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
impl<T> Dot<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.x),
                    std::intrinsics::fmul_fast(self.y, rhs.y),
                ),
                std::intrinsics::fmul_fast(self.z, rhs.z),
            )
        }
    }
}
impl<T> Dot<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.y, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.xz),
                ),
                y: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.yz),
                ),
                z: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xz),
                    std::intrinsics::fmul_fast(self.y, rhs.yz),
                ),
            }
        }
    }
}
impl<T> Dot<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.z, rhs.xyz),
                xz: -std::intrinsics::fmul_fast(self.y, rhs.xyz),
                yz: std::intrinsics::fmul_fast(self.x, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0>
    Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: ScalarProduct<G1Rhs, Output = G0_0>,
    Vector<T>: VectorProduct<G0Rhs, Output = G1_0>,
    Vector<T>: VectorProduct<G2Rhs, Output = G1_1>,
    Vector<T>: BivectorProduct<G3Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.1),
            self.vector_prod(rhs.0) + self.vector_prod(rhs.2),
            self.bivector_prod(rhs.3),
            Zero,
        )
    }
}
impl<T> Dot<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Dot<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> Dot<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.xy, rhs.y),
                    std::intrinsics::fmul_fast(self.xz, rhs.z),
                ),
                y: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.x),
                    std::intrinsics::fmul_fast(self.yz, rhs.z),
                ),
                z: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xz, rhs.x),
                    -std::intrinsics::fmul_fast(self.yz, rhs.y),
                ),
            }
        }
    }
}
impl<T> Dot<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.xy),
                    -std::intrinsics::fmul_fast(self.xz, rhs.xz),
                ),
                -std::intrinsics::fmul_fast(self.yz, rhs.yz),
            )
        }
    }
}
impl<T> Dot<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.yz, rhs.xyz),
                y: std::intrinsics::fmul_fast(self.xz, rhs.xyz),
                z: -std::intrinsics::fmul_fast(self.xy, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G1_1, G1_2, G2_0>
    Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: ScalarProduct<G2Rhs, Output = G0_0>,
    Bivector<T>: VectorProduct<G1Rhs, Output = G1_0>,
    Bivector<T>: VectorProduct<G3Rhs, Output = G1_1>,
    Bivector<T>: BivectorProduct<G0Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.2),
            self.vector_prod(rhs.1) + self.vector_prod(rhs.3),
            self.bivector_prod(rhs.0),
            Zero,
        )
    }
}
impl<T> Dot<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Dot<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T> Dot<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xyz, rhs.z),
                xz: -std::intrinsics::fmul_fast(self.xyz, rhs.y),
                yz: std::intrinsics::fmul_fast(self.xyz, rhs.x),
            }
        }
    }
}
impl<T> Dot<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.xyz, rhs.yz),
                y: std::intrinsics::fmul_fast(self.xyz, rhs.xz),
                z: -std::intrinsics::fmul_fast(self.xyz, rhs.xy),
            }
        }
    }
}
impl<T> Dot<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: ScalarProduct<G3Rhs, Output = G0_0>,
    Trivector<T>: VectorProduct<G2Rhs, Output = G1_0>,
    Trivector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    Trivector<T>: TrivectorProduct<G0Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.3),
            self.vector_prod(rhs.2),
            self.bivector_prod(rhs.1),
            self.trivector_prod(rhs.0),
        )
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> Dot<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> Dot<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: ScalarProduct<T, Output = G0_0>,
    G1Lhs: VectorProduct<T, Output = G1_0>,
    G2Lhs: BivectorProduct<T, Output = G2_0>,
    G3Lhs: TrivectorProduct<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: T) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs),
            self.1.vector_prod(rhs),
            self.2.bivector_prod(rhs),
            self.3.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0> Dot<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: ScalarProduct<Vector<T>, Output = G0_0>,
    G0Lhs: VectorProduct<Vector<T>, Output = G1_0>,
    G2Lhs: VectorProduct<Vector<T>, Output = G1_1>,
    G3Lhs: BivectorProduct<Vector<T>, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Vector<T>) -> Self::Output {
        Multivector(
            self.1.scalar_prod(rhs),
            self.0.vector_prod(rhs) + self.2.vector_prod(rhs),
            self.3.bivector_prod(rhs),
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G1_1, G1_2, G2_0> Dot<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: ScalarProduct<Bivector<T>, Output = G0_0>,
    G1Lhs: VectorProduct<Bivector<T>, Output = G1_0>,
    G3Lhs: VectorProduct<Bivector<T>, Output = G1_1>,
    G0Lhs: BivectorProduct<Bivector<T>, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = Multivector<G0_0, G1_2, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(
            self.2.scalar_prod(rhs),
            self.1.vector_prod(rhs) + self.3.vector_prod(rhs),
            self.0.bivector_prod(rhs),
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> Dot<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: ScalarProduct<Trivector<T>, Output = G0_0>,
    G2Lhs: VectorProduct<Trivector<T>, Output = G1_0>,
    G1Lhs: BivectorProduct<Trivector<T>, Output = G2_0>,
    G0Lhs: TrivectorProduct<Trivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(
            self.3.scalar_prod(rhs),
            self.2.vector_prod(rhs),
            self.1.bivector_prod(rhs),
            self.0.trivector_prod(rhs),
        )
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G1_5,
        G1_6,
        G1_7,
        G1_8,
        G1_9,
        G1_10,
        G2_0,
        G2_1,
        G2_2,
        G2_3,
        G2_4,
        G2_5,
        G2_6,
        G3_0,
        G3_1,
        G3_2,
    > Dot<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ScalarProduct<G0Rhs, Output = G0_0>,
    G1Lhs: ScalarProduct<G1Rhs, Output = G0_1>,
    G2Lhs: ScalarProduct<G2Rhs, Output = G0_2>,
    G3Lhs: ScalarProduct<G3Rhs, Output = G0_3>,
    G0Lhs: VectorProduct<G1Rhs, Output = G1_0>,
    G1Lhs: VectorProduct<G0Rhs, Output = G1_1>,
    G1Lhs: VectorProduct<G2Rhs, Output = G1_2>,
    G2Lhs: VectorProduct<G1Rhs, Output = G1_3>,
    G2Lhs: VectorProduct<G3Rhs, Output = G1_4>,
    G3Lhs: VectorProduct<G2Rhs, Output = G1_5>,
    G0Lhs: BivectorProduct<G2Rhs, Output = G2_0>,
    G1Lhs: BivectorProduct<G3Rhs, Output = G2_1>,
    G2Lhs: BivectorProduct<G0Rhs, Output = G2_2>,
    G3Lhs: BivectorProduct<G1Rhs, Output = G2_3>,
    G0Lhs: TrivectorProduct<G3Rhs, Output = G3_0>,
    G3Lhs: TrivectorProduct<G0Rhs, Output = G3_1>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
    G1_0: std::ops::Add<G1_1, Output = G1_6>,
    G1_6: std::ops::Add<G1_2, Output = G1_7>,
    G1_7: std::ops::Add<G1_3, Output = G1_8>,
    G1_8: std::ops::Add<G1_4, Output = G1_9>,
    G1_9: std::ops::Add<G1_5, Output = G1_10>,
    G2_0: std::ops::Add<G2_1, Output = G2_4>,
    G2_4: std::ops::Add<G2_2, Output = G2_5>,
    G2_5: std::ops::Add<G2_3, Output = G2_6>,
    G3_0: std::ops::Add<G3_1, Output = G3_2>,
{
    type Output = Multivector<G0_6, G1_10, G2_6, G3_2>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn dot(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs.0)
                + self.1.scalar_prod(rhs.1)
                + self.2.scalar_prod(rhs.2)
                + self.3.scalar_prod(rhs.3),
            self.0.vector_prod(rhs.1)
                + self.1.vector_prod(rhs.0)
                + self.1.vector_prod(rhs.2)
                + self.2.vector_prod(rhs.1)
                + self.2.vector_prod(rhs.3)
                + self.3.vector_prod(rhs.2),
            self.0.bivector_prod(rhs.2)
                + self.1.bivector_prod(rhs.3)
                + self.2.bivector_prod(rhs.0)
                + self.3.bivector_prod(rhs.1),
            self.0.trivector_prod(rhs.3) + self.3.trivector_prod(rhs.0),
        )
    }
}
impl Wedge<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl Wedge<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl Wedge<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl Wedge<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl Wedge<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl Wedge<Vector<f32>> for f32 {
    type Output = Vector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<f32>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl Wedge<Vector<f64>> for f64 {
    type Output = Vector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<f64>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl Wedge<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Wedge<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl Wedge<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl Wedge<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: ScalarProduct<G0Rhs, Output = G0_0>,
    f32: VectorProduct<G1Rhs, Output = G1_0>,
    f32: BivectorProduct<G2Rhs, Output = G2_0>,
    f32: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: ScalarProduct<G0Rhs, Output = G0_0>,
    f64: VectorProduct<G1Rhs, Output = G1_0>,
    f64: BivectorProduct<G2Rhs, Output = G2_0>,
    f64: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.scalar_prod(rhs.0),
            self.vector_prod(rhs.1),
            self.bivector_prod(rhs.2),
            self.trivector_prod(rhs.3),
        )
    }
}
impl<T> Wedge<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
impl<T> Wedge<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.y),
                    -std::intrinsics::fmul_fast(self.y, rhs.x),
                ),
                xz: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.z),
                    -std::intrinsics::fmul_fast(self.z, rhs.x),
                ),
                yz: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.y, rhs.z),
                    -std::intrinsics::fmul_fast(self.z, rhs.y),
                ),
            }
        }
    }
}
impl<T> Wedge<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(
                    std::intrinsics::fadd_fast(
                        std::intrinsics::fmul_fast(self.x, rhs.yz),
                        -std::intrinsics::fmul_fast(self.y, rhs.xz),
                    ),
                    std::intrinsics::fmul_fast(self.z, rhs.xy),
                ),
            }
        }
    }
}
impl<T> Wedge<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0, G2_0, G3_0> Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: VectorProduct<G0Rhs, Output = G1_0>,
    Vector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    Vector<T>: TrivectorProduct<G2Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<Zero, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            Zero,
            self.vector_prod(rhs.0),
            self.bivector_prod(rhs.1),
            self.trivector_prod(rhs.2),
        )
    }
}
impl<T> Wedge<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> Wedge<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(
                    std::intrinsics::fadd_fast(
                        std::intrinsics::fmul_fast(self.xy, rhs.z),
                        -std::intrinsics::fmul_fast(self.xz, rhs.y),
                    ),
                    std::intrinsics::fmul_fast(self.yz, rhs.x),
                ),
            }
        }
    }
}
impl<T> Wedge<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0, G3_0> Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: BivectorProduct<G0Rhs, Output = G2_0>,
    Bivector<T>: TrivectorProduct<G1Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<Zero, Zero, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            Zero,
            Zero,
            self.bivector_prod(rhs.0),
            self.trivector_prod(rhs.1),
        )
    }
}
impl<T> Wedge<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T> Wedge<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> Wedge<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: TrivectorProduct<G0Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.0)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> Wedge<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> Wedge<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: ScalarProduct<T, Output = G0_0>,
    G1Lhs: VectorProduct<T, Output = G1_0>,
    G2Lhs: BivectorProduct<T, Output = G2_0>,
    G3Lhs: TrivectorProduct<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: T) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs),
            self.1.vector_prod(rhs),
            self.2.bivector_prod(rhs),
            self.3.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1_0, G2_0, G3_0> Wedge<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: Copy,
    G0Lhs: VectorProduct<Vector<T>, Output = G1_0>,
    G1Lhs: BivectorProduct<Vector<T>, Output = G2_0>,
    G2Lhs: TrivectorProduct<Vector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<Zero, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Vector<T>) -> Self::Output {
        Multivector(
            Zero,
            self.0.vector_prod(rhs),
            self.1.bivector_prod(rhs),
            self.2.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2_0, G3_0> Wedge<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: BivectorProduct<Bivector<T>, Output = G2_0>,
    G1Lhs: TrivectorProduct<Bivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<Zero, Zero, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(
            Zero,
            Zero,
            self.0.bivector_prod(rhs),
            self.1.trivector_prod(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3_0> Wedge<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: TrivectorProduct<Trivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Trivector<T>) -> Self::Output {
        self.0.trivector_prod(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G1_0,
        G1_1,
        G1_2,
        G2_0,
        G2_1,
        G2_2,
        G2_3,
        G2_4,
        G3_0,
        G3_1,
        G3_2,
        G3_3,
        G3_4,
        G3_5,
        G3_6,
    > Wedge<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ScalarProduct<G0Rhs, Output = G0_0>,
    G0Lhs: VectorProduct<G1Rhs, Output = G1_0>,
    G1Lhs: VectorProduct<G0Rhs, Output = G1_1>,
    G0Lhs: BivectorProduct<G2Rhs, Output = G2_0>,
    G1Lhs: BivectorProduct<G1Rhs, Output = G2_1>,
    G2Lhs: BivectorProduct<G0Rhs, Output = G2_2>,
    G0Lhs: TrivectorProduct<G3Rhs, Output = G3_0>,
    G1Lhs: TrivectorProduct<G2Rhs, Output = G3_1>,
    G2Lhs: TrivectorProduct<G1Rhs, Output = G3_2>,
    G3Lhs: TrivectorProduct<G0Rhs, Output = G3_3>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
    G2_0: std::ops::Add<G2_1, Output = G2_3>,
    G2_3: std::ops::Add<G2_2, Output = G2_4>,
    G3_0: std::ops::Add<G3_1, Output = G3_4>,
    G3_4: std::ops::Add<G3_2, Output = G3_5>,
    G3_5: std::ops::Add<G3_3, Output = G3_6>,
{
    type Output = Multivector<G0_0, G1_2, G2_4, G3_6>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn wedge(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.scalar_prod(rhs.0),
            self.0.vector_prod(rhs.1) + self.1.vector_prod(rhs.0),
            self.0.bivector_prod(rhs.2) + self.1.bivector_prod(rhs.1) + self.2.bivector_prod(rhs.0),
            self.0.trivector_prod(rhs.3)
                + self.1.trivector_prod(rhs.2)
                + self.2.trivector_prod(rhs.1)
                + self.3.trivector_prod(rhs.0),
        )
    }
}
impl LeftContraction<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl LeftContraction<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl LeftContraction<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl LeftContraction<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl LeftContraction<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl LeftContraction<Vector<f32>> for f32 {
    type Output = Vector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<f32>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl LeftContraction<Vector<f64>> for f64 {
    type Output = Vector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<f64>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self, rhs.x),
                y: std::intrinsics::fmul_fast(self, rhs.y),
                z: std::intrinsics::fmul_fast(self, rhs.z),
            }
        }
    }
}
impl LeftContraction<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl LeftContraction<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl LeftContraction<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl LeftContraction<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
where
    f32: LeftContraction<G0Rhs, Output = G0_0>,
    f32: LeftContraction<G1Rhs, Output = G1_0>,
    f32: LeftContraction<G2Rhs, Output = G2_0>,
    f32: LeftContraction<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.left_contraction(rhs.0),
            self.left_contraction(rhs.1),
            self.left_contraction(rhs.2),
            self.left_contraction(rhs.3),
        )
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f64
where
    f64: LeftContraction<G0Rhs, Output = G0_0>,
    f64: LeftContraction<G1Rhs, Output = G1_0>,
    f64: LeftContraction<G2Rhs, Output = G2_0>,
    f64: LeftContraction<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.left_contraction(rhs.0),
            self.left_contraction(rhs.1),
            self.left_contraction(rhs.2),
            self.left_contraction(rhs.3),
        )
    }
}
impl<T> LeftContraction<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.x),
                    std::intrinsics::fmul_fast(self.y, rhs.y),
                ),
                std::intrinsics::fmul_fast(self.z, rhs.z),
            )
        }
    }
}
impl<T> LeftContraction<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.y, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.xz),
                ),
                y: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.yz),
                ),
                z: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xz),
                    std::intrinsics::fmul_fast(self.y, rhs.yz),
                ),
            }
        }
    }
}
impl<T> LeftContraction<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.z, rhs.xyz),
                xz: -std::intrinsics::fmul_fast(self.y, rhs.xyz),
                yz: std::intrinsics::fmul_fast(self.x, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0>
    LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: LeftContraction<G1Rhs, Output = G0_0>,
    Vector<T>: LeftContraction<G2Rhs, Output = G1_0>,
    Vector<T>: LeftContraction<G3Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.left_contraction(rhs.1),
            self.left_contraction(rhs.2),
            self.left_contraction(rhs.3),
            Zero,
        )
    }
}
impl<T> LeftContraction<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.xy),
                    -std::intrinsics::fmul_fast(self.xz, rhs.xz),
                ),
                -std::intrinsics::fmul_fast(self.yz, rhs.yz),
            )
        }
    }
}
impl<T> LeftContraction<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.yz, rhs.xyz),
                y: std::intrinsics::fmul_fast(self.xz, rhs.xyz),
                z: -std::intrinsics::fmul_fast(self.xy, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0>
    LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: LeftContraction<G2Rhs, Output = G0_0>,
    Bivector<T>: LeftContraction<G3Rhs, Output = G1_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.left_contraction(rhs.2),
            self.left_contraction(rhs.3),
            Zero,
            Zero,
        )
    }
}
impl<T> LeftContraction<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> LeftContraction<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: LeftContraction<G3Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.left_contraction(rhs.3)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> LeftContraction<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> LeftContraction<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: LeftContraction<T, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: T) -> Self::Output {
        self.0.left_contraction(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0> LeftContraction<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: LeftContraction<Vector<T>, Output = G0_0>,
    G0Lhs: LeftContraction<Vector<T>, Output = G1_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Vector<T>) -> Self::Output {
        Multivector(
            self.1.left_contraction(rhs),
            self.0.left_contraction(rhs),
            Zero,
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0> LeftContraction<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: LeftContraction<Bivector<T>, Output = G0_0>,
    G1Lhs: LeftContraction<Bivector<T>, Output = G1_0>,
    G0Lhs: LeftContraction<Bivector<T>, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(
            self.2.left_contraction(rhs),
            self.1.left_contraction(rhs),
            self.0.left_contraction(rhs),
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> LeftContraction<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: LeftContraction<Trivector<T>, Output = G0_0>,
    G2Lhs: LeftContraction<Trivector<T>, Output = G1_0>,
    G1Lhs: LeftContraction<Trivector<T>, Output = G2_0>,
    G0Lhs: LeftContraction<Trivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Trivector<T>) -> Self::Output {
        Multivector(
            self.3.left_contraction(rhs),
            self.2.left_contraction(rhs),
            self.1.left_contraction(rhs),
            self.0.left_contraction(rhs),
        )
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G2_0,
        G2_1,
        G2_2,
        G3_0,
    > LeftContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: LeftContraction<G0Rhs, Output = G0_0>,
    G1Lhs: LeftContraction<G1Rhs, Output = G0_1>,
    G2Lhs: LeftContraction<G2Rhs, Output = G0_2>,
    G3Lhs: LeftContraction<G3Rhs, Output = G0_3>,
    G0Lhs: LeftContraction<G1Rhs, Output = G1_0>,
    G1Lhs: LeftContraction<G2Rhs, Output = G1_1>,
    G2Lhs: LeftContraction<G3Rhs, Output = G1_2>,
    G0Lhs: LeftContraction<G2Rhs, Output = G2_0>,
    G1Lhs: LeftContraction<G3Rhs, Output = G2_1>,
    G0Lhs: LeftContraction<G3Rhs, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
    G1_0: std::ops::Add<G1_1, Output = G1_3>,
    G1_3: std::ops::Add<G1_2, Output = G1_4>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_6, G1_4, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn left_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.left_contraction(rhs.0)
                + self.1.left_contraction(rhs.1)
                + self.2.left_contraction(rhs.2)
                + self.3.left_contraction(rhs.3),
            self.0.left_contraction(rhs.1)
                + self.1.left_contraction(rhs.2)
                + self.2.left_contraction(rhs.3),
            self.0.left_contraction(rhs.2) + self.1.left_contraction(rhs.3),
            self.0.left_contraction(rhs.3),
        )
    }
}
impl RightContraction<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl RightContraction<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl RightContraction<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl RightContraction<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl RightContraction<Vector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<f32>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Vector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<f64>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Bivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<f32>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Bivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<f64>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Trivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<f32>) -> Self::Output {
        Zero
    }
}
impl RightContraction<Trivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<f64>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: RightContraction<G0Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.right_contraction(rhs.0)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: RightContraction<G0Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.right_contraction(rhs.0)
    }
}
impl<T> RightContraction<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
impl<T> RightContraction<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.x),
                    std::intrinsics::fmul_fast(self.y, rhs.y),
                ),
                std::intrinsics::fmul_fast(self.z, rhs.z),
            )
        }
    }
}
impl<T> RightContraction<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0>
    RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: RightContraction<G1Rhs, Output = G0_0>,
    Vector<T>: RightContraction<G0Rhs, Output = G1_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.right_contraction(rhs.1),
            self.right_contraction(rhs.0),
            Zero,
            Zero,
        )
    }
}
impl<T> RightContraction<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> RightContraction<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.xy, rhs.y),
                    std::intrinsics::fmul_fast(self.xz, rhs.z),
                ),
                y: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.x),
                    std::intrinsics::fmul_fast(self.yz, rhs.z),
                ),
                z: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xz, rhs.x),
                    -std::intrinsics::fmul_fast(self.yz, rhs.y),
                ),
            }
        }
    }
}
impl<T> RightContraction<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.xy),
                    -std::intrinsics::fmul_fast(self.xz, rhs.xz),
                ),
                -std::intrinsics::fmul_fast(self.yz, rhs.yz),
            )
        }
    }
}
impl<T> RightContraction<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0>
    RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: RightContraction<G2Rhs, Output = G0_0>,
    Bivector<T>: RightContraction<G1Rhs, Output = G1_0>,
    Bivector<T>: RightContraction<G0Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.right_contraction(rhs.2),
            self.right_contraction(rhs.1),
            self.right_contraction(rhs.0),
            Zero,
        )
    }
}
impl<T> RightContraction<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> RightContraction<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T> RightContraction<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xyz, rhs.z),
                xz: -std::intrinsics::fmul_fast(self.xyz, rhs.y),
                yz: std::intrinsics::fmul_fast(self.xyz, rhs.x),
            }
        }
    }
}
impl<T> RightContraction<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.xyz, rhs.yz),
                y: std::intrinsics::fmul_fast(self.xyz, rhs.xz),
                z: -std::intrinsics::fmul_fast(self.xyz, rhs.xy),
            }
        }
    }
}
impl<T> RightContraction<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
    RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: RightContraction<G3Rhs, Output = G0_0>,
    Trivector<T>: RightContraction<G2Rhs, Output = G1_0>,
    Trivector<T>: RightContraction<G1Rhs, Output = G2_0>,
    Trivector<T>: RightContraction<G0Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.right_contraction(rhs.3),
            self.right_contraction(rhs.2),
            self.right_contraction(rhs.1),
            self.right_contraction(rhs.0),
        )
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> RightContraction<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> RightContraction<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: RightContraction<T, Output = G0_0>,
    G1Lhs: RightContraction<T, Output = G1_0>,
    G2Lhs: RightContraction<T, Output = G2_0>,
    G3Lhs: RightContraction<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: T) -> Self::Output {
        Multivector(
            self.0.right_contraction(rhs),
            self.1.right_contraction(rhs),
            self.2.right_contraction(rhs),
            self.3.right_contraction(rhs),
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0> RightContraction<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: RightContraction<Vector<T>, Output = G0_0>,
    G2Lhs: RightContraction<Vector<T>, Output = G1_0>,
    G3Lhs: RightContraction<Vector<T>, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Vector<T>) -> Self::Output {
        Multivector(
            self.1.right_contraction(rhs),
            self.2.right_contraction(rhs),
            self.3.right_contraction(rhs),
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0> RightContraction<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: RightContraction<Bivector<T>, Output = G0_0>,
    G3Lhs: RightContraction<Bivector<T>, Output = G1_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, Zero, Zero>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Bivector<T>) -> Self::Output {
        Multivector(
            self.2.right_contraction(rhs),
            self.3.right_contraction(rhs),
            Zero,
            Zero,
        )
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> RightContraction<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: RightContraction<Trivector<T>, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Trivector<T>) -> Self::Output {
        self.3.right_contraction(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G2_0,
        G2_1,
        G2_2,
        G3_0,
    > RightContraction<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: RightContraction<G0Rhs, Output = G0_0>,
    G1Lhs: RightContraction<G1Rhs, Output = G0_1>,
    G2Lhs: RightContraction<G2Rhs, Output = G0_2>,
    G3Lhs: RightContraction<G3Rhs, Output = G0_3>,
    G1Lhs: RightContraction<G0Rhs, Output = G1_0>,
    G2Lhs: RightContraction<G1Rhs, Output = G1_1>,
    G3Lhs: RightContraction<G2Rhs, Output = G1_2>,
    G2Lhs: RightContraction<G0Rhs, Output = G2_0>,
    G3Lhs: RightContraction<G1Rhs, Output = G2_1>,
    G3Lhs: RightContraction<G0Rhs, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
    G1_0: std::ops::Add<G1_1, Output = G1_3>,
    G1_3: std::ops::Add<G1_2, Output = G1_4>,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = Multivector<G0_6, G1_4, G2_2, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn right_contraction(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Multivector(
            self.0.right_contraction(rhs.0)
                + self.1.right_contraction(rhs.1)
                + self.2.right_contraction(rhs.2)
                + self.3.right_contraction(rhs.3),
            self.1.right_contraction(rhs.0)
                + self.2.right_contraction(rhs.1)
                + self.3.right_contraction(rhs.2),
            self.2.right_contraction(rhs.0) + self.3.right_contraction(rhs.1),
            self.3.right_contraction(rhs.0),
        )
    }
}
impl<T> std::ops::Div<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn div(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Div<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn div(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fdiv_fast(self.x, rhs),
                y: std::intrinsics::fdiv_fast(self.y, rhs),
                z: std::intrinsics::fdiv_fast(self.z, rhs),
            }
        }
    }
}
impl<T> std::ops::Div<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn div(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fdiv_fast(self.xy, rhs),
                xz: std::intrinsics::fdiv_fast(self.xz, rhs),
                yz: std::intrinsics::fdiv_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> std::ops::Div<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn div(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fdiv_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0, G1_0, G2_0, G3_0> std::ops::Div<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: std::ops::Div<T, Output = G0_0>,
    G1Lhs: std::ops::Div<T, Output = G1_0>,
    G2Lhs: std::ops::Div<T, Output = G2_0>,
    G3Lhs: std::ops::Div<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn div(self, rhs: T) -> Self::Output {
        Multivector(
            self.0.div(rhs),
            self.1.div(rhs),
            self.2.div(rhs),
            self.3.div(rhs),
        )
    }
}
impl ScalarProduct<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<f32> for f32 {
    type Output = f32;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: f32) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl ScalarProduct<f64> for f64 {
    type Output = f64;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: f64) -> Self::Output {
        unsafe { std::intrinsics::fmul_fast(self, rhs) }
    }
}
impl ScalarProduct<Vector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<f32>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Vector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<f64>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Bivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<f32>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Bivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<f64>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Trivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<f32>) -> Self::Output {
        Zero
    }
}
impl ScalarProduct<Trivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<f64>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: ScalarProduct<G0Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.scalar_prod(rhs.0)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: ScalarProduct<G0Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.scalar_prod(rhs.0)
    }
}
impl<T> ScalarProduct<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.x),
                    std::intrinsics::fmul_fast(self.y, rhs.y),
                ),
                std::intrinsics::fmul_fast(self.z, rhs.z),
            )
        }
    }
}
impl<T> ScalarProduct<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: ScalarProduct<G1Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.scalar_prod(rhs.1)
    }
}
impl<T> ScalarProduct<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            std::intrinsics::fadd_fast(
                std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.xy),
                    -std::intrinsics::fmul_fast(self.xz, rhs.xz),
                ),
                -std::intrinsics::fmul_fast(self.yz, rhs.yz),
            )
        }
    }
}
impl<T> ScalarProduct<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: ScalarProduct<G2Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.scalar_prod(rhs.2)
    }
}
impl<T> ScalarProduct<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> ScalarProduct<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<T>) -> Self::Output {
        unsafe { -std::intrinsics::fmul_fast(self.xyz, rhs.xyz) }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: ScalarProduct<G3Rhs, Output = G0_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.scalar_prod(rhs.3)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> ScalarProduct<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> ScalarProduct<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: ScalarProduct<T, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: T) -> Self::Output {
        self.0.scalar_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> ScalarProduct<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: ScalarProduct<Vector<T>, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Vector<T>) -> Self::Output {
        self.1.scalar_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> ScalarProduct<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: ScalarProduct<Bivector<T>, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Bivector<T>) -> Self::Output {
        self.2.scalar_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G0_0> ScalarProduct<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: ScalarProduct<Trivector<T>, Output = G0_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G0_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Trivector<T>) -> Self::Output {
        self.3.scalar_prod(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G0_0,
        G0_1,
        G0_2,
        G0_3,
        G0_4,
        G0_5,
        G0_6,
    > ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ScalarProduct<G0Rhs, Output = G0_0>,
    G1Lhs: ScalarProduct<G1Rhs, Output = G0_1>,
    G2Lhs: ScalarProduct<G2Rhs, Output = G0_2>,
    G3Lhs: ScalarProduct<G3Rhs, Output = G0_3>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G0_0: std::ops::Add<G0_1, Output = G0_4>,
    G0_4: std::ops::Add<G0_2, Output = G0_5>,
    G0_5: std::ops::Add<G0_3, Output = G0_6>,
{
    type Output = G0_6;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.0.scalar_prod(rhs.0)
            + self.1.scalar_prod(rhs.1)
            + self.2.scalar_prod(rhs.2)
            + self.3.scalar_prod(rhs.3)
    }
}
impl VectorProduct<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl VectorProduct<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl VectorProduct<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T: Float> VectorProduct<T> for T {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
// impl VectorProduct<f32> for f32 {
//     type Output = Zero;
//     #[inline]
//         #[allow(unused_variables, unused_unsafe)]
//         fn vector_prod(self, rhs: f32) -> Self::Output {
//         Zero
//     }
// }
// impl VectorProduct<f64> for f64 {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn vector_prod(self, rhs: f64) -> Self::Output {
//         Zero
//     }
// }
impl<T: Float> VectorProduct<Vector<T>> for T {
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: self * rhs.x,
                y: self * rhs.y,
                z: self * rhs.z,
            }
        }
    }
}
// impl VectorProduct<Vector<f32>> for f32 {
//     type Output = Vector<f32>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn vector_prod(self, rhs: Vector<f32>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: std::intrinsics::fmul_fast(self, rhs.x),
//                 y: std::intrinsics::fmul_fast(self, rhs.y),
//                 z: std::intrinsics::fmul_fast(self, rhs.z),
//             }
//         }
//     }
// }
// impl VectorProduct<Vector<f64>> for f64 {
//     type Output = Vector<f64>;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn vector_prod(self, rhs: Vector<f64>) -> Self::Output {
//         unsafe {
//             Vector {
//                 x: std::intrinsics::fmul_fast(self, rhs.x),
//                 y: std::intrinsics::fmul_fast(self, rhs.y),
//                 z: std::intrinsics::fmul_fast(self, rhs.z),
//             }
//         }
//     }
// }
impl<T: Float> VectorProduct<Bivector<T>> for T {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
// impl VectorProduct<Bivector<f32>> for f32 {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn vector_prod(self, rhs: Bivector<f32>) -> Self::Output {
//         Zero
//     }
// }
// impl VectorProduct<Bivector<f64>> for f64 {
//     type Output = Zero;
//     #[inline]
//     #[allow(unused_variables, unused_unsafe)]
//     fn vector_prod(self, rhs: Bivector<f64>) -> Self::Output {
//         Zero
//     }
// }
impl VectorProduct<Trivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<f32>) -> Self::Output {
        Zero
    }
}
impl VectorProduct<Trivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<f64>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0> VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: VectorProduct<G1Rhs, Output = G1_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G1_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.vector_prod(rhs.1)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0> VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: VectorProduct<G1Rhs, Output = G1_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G1_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.vector_prod(rhs.1)
    }
}
impl<T> VectorProduct<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fmul_fast(self.x, rhs),
                y: std::intrinsics::fmul_fast(self.y, rhs),
                z: std::intrinsics::fmul_fast(self.z, rhs),
            }
        }
    }
}
impl<T> VectorProduct<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.y, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.xz),
                ),
                y: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xy),
                    -std::intrinsics::fmul_fast(self.z, rhs.yz),
                ),
                z: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.xz),
                    std::intrinsics::fmul_fast(self.y, rhs.yz),
                ),
            }
        }
    }
}
impl<T> VectorProduct<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0, G1_1, G1_2>
    VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: VectorProduct<G0Rhs, Output = G1_0>,
    Vector<T>: VectorProduct<G2Rhs, Output = G1_1>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = G1_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.vector_prod(rhs.0) + self.vector_prod(rhs.2)
    }
}
impl<T> VectorProduct<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.xy, rhs.y),
                    std::intrinsics::fmul_fast(self.xz, rhs.z),
                ),
                y: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.x),
                    std::intrinsics::fmul_fast(self.yz, rhs.z),
                ),
                z: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xz, rhs.x),
                    -std::intrinsics::fmul_fast(self.yz, rhs.y),
                ),
            }
        }
    }
}
impl<T> VectorProduct<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Trivector<T>> for Bivector<T>
where
    T: Copy + num_traits::NumOps + std::ops::Neg<Output = T>,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: self.yz * rhs.xyz,
                y: self.xz * rhs.xyz,
                z: -self.xy * rhs.xyz,
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0, G1_1, G1_2>
    VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: VectorProduct<G1Rhs, Output = G1_0>,
    Bivector<T>: VectorProduct<G3Rhs, Output = G1_1>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = G1_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.vector_prod(rhs.1) + self.vector_prod(rhs.3)
    }
}
impl<T> VectorProduct<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> VectorProduct<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Vector {
                x: -std::intrinsics::fmul_fast(self.xyz, rhs.yz),
                y: std::intrinsics::fmul_fast(self.xyz, rhs.xz),
                z: -std::intrinsics::fmul_fast(self.xyz, rhs.xy),
            }
        }
    }
}
impl<T> VectorProduct<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G1_0> VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: VectorProduct<G2Rhs, Output = G1_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G1_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.vector_prod(rhs.2)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> VectorProduct<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1_0> VectorProduct<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: VectorProduct<T, Output = G1_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G1_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: T) -> Self::Output {
        self.1.vector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1_0, G1_1, G1_2> VectorProduct<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: VectorProduct<Vector<T>, Output = G1_0>,
    G2Lhs: VectorProduct<Vector<T>, Output = G1_1>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = G1_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Vector<T>) -> Self::Output {
        self.0.vector_prod(rhs) + self.2.vector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1_0, G1_1, G1_2> VectorProduct<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: VectorProduct<Bivector<T>, Output = G1_0>,
    G3Lhs: VectorProduct<Bivector<T>, Output = G1_1>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_2>,
{
    type Output = G1_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Bivector<T>) -> Self::Output {
        self.1.vector_prod(rhs) + self.3.vector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G1_0> VectorProduct<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: VectorProduct<Trivector<T>, Output = G1_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G1_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Trivector<T>) -> Self::Output {
        self.2.vector_prod(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G1_0,
        G1_1,
        G1_2,
        G1_3,
        G1_4,
        G1_5,
        G1_6,
        G1_7,
        G1_8,
        G1_9,
        G1_10,
    > VectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: VectorProduct<G1Rhs, Output = G1_0>,
    G1Lhs: VectorProduct<G0Rhs, Output = G1_1>,
    G1Lhs: VectorProduct<G2Rhs, Output = G1_2>,
    G2Lhs: VectorProduct<G1Rhs, Output = G1_3>,
    G2Lhs: VectorProduct<G3Rhs, Output = G1_4>,
    G3Lhs: VectorProduct<G2Rhs, Output = G1_5>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G1_0: std::ops::Add<G1_1, Output = G1_6>,
    G1_6: std::ops::Add<G1_2, Output = G1_7>,
    G1_7: std::ops::Add<G1_3, Output = G1_8>,
    G1_8: std::ops::Add<G1_4, Output = G1_9>,
    G1_9: std::ops::Add<G1_5, Output = G1_10>,
{
    type Output = G1_10;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn vector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.0.vector_prod(rhs.1)
            + self.1.vector_prod(rhs.0)
            + self.1.vector_prod(rhs.2)
            + self.2.vector_prod(rhs.1)
            + self.2.vector_prod(rhs.3)
            + self.3.vector_prod(rhs.2)
    }
}
impl BivectorProduct<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<f32> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: f32) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<f64> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: f64) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Vector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<f32>) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Vector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<f64>) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Bivector<f32>> for f32 {
    type Output = Bivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<f32>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl BivectorProduct<Bivector<f64>> for f64 {
    type Output = Bivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<f64>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self, rhs.xy),
                xz: std::intrinsics::fmul_fast(self, rhs.xz),
                yz: std::intrinsics::fmul_fast(self, rhs.yz),
            }
        }
    }
}
impl BivectorProduct<Trivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<f32>) -> Self::Output {
        Zero
    }
}
impl BivectorProduct<Trivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<f64>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0> BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: BivectorProduct<G2Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G2_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.bivector_prod(rhs.2)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0> BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: BivectorProduct<G2Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G2_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.bivector_prod(rhs.2)
    }
}
impl<T> BivectorProduct<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.y),
                    -std::intrinsics::fmul_fast(self.y, rhs.x),
                ),
                xz: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.x, rhs.z),
                    -std::intrinsics::fmul_fast(self.z, rhs.x),
                ),
                yz: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.y, rhs.z),
                    -std::intrinsics::fmul_fast(self.z, rhs.y),
                ),
            }
        }
    }
}
impl<T> BivectorProduct<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.z, rhs.xyz),
                xz: -std::intrinsics::fmul_fast(self.y, rhs.xyz),
                yz: std::intrinsics::fmul_fast(self.x, rhs.xyz),
            }
        }
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0, G2_1, G2_2>
    BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    Vector<T>: BivectorProduct<G3Rhs, Output = G2_1>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = G2_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.bivector_prod(rhs.1) + self.bivector_prod(rhs.3)
    }
}
impl<T> BivectorProduct<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: T) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xy, rhs),
                xz: std::intrinsics::fmul_fast(self.xz, rhs),
                yz: std::intrinsics::fmul_fast(self.yz, rhs),
            }
        }
    }
}
impl<T> BivectorProduct<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xz, rhs.yz),
                    std::intrinsics::fmul_fast(self.yz, rhs.xz),
                ),
                xz: std::intrinsics::fadd_fast(
                    std::intrinsics::fmul_fast(self.xy, rhs.yz),
                    -std::intrinsics::fmul_fast(self.yz, rhs.xy),
                ),
                yz: std::intrinsics::fadd_fast(
                    -std::intrinsics::fmul_fast(self.xy, rhs.xz),
                    std::intrinsics::fmul_fast(self.xz, rhs.xy),
                ),
            }
        }
    }
}
impl<T> BivectorProduct<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0, G2_1, G2_2>
    BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: BivectorProduct<G0Rhs, Output = G2_0>,
    Bivector<T>: BivectorProduct<G2Rhs, Output = G2_1>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = G2_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.bivector_prod(rhs.0) + self.bivector_prod(rhs.2)
    }
}
impl<T> BivectorProduct<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Bivector {
                xy: std::intrinsics::fmul_fast(self.xyz, rhs.z),
                xz: -std::intrinsics::fmul_fast(self.xyz, rhs.y),
                yz: std::intrinsics::fmul_fast(self.xyz, rhs.x),
            }
        }
    }
}
impl<T> BivectorProduct<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> BivectorProduct<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G2_0> BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: BivectorProduct<G1Rhs, Output = G2_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G2_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.bivector_prod(rhs.1)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> BivectorProduct<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2_0> BivectorProduct<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: BivectorProduct<T, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G2_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: T) -> Self::Output {
        self.2.bivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2_0, G2_1, G2_2> BivectorProduct<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: BivectorProduct<Vector<T>, Output = G2_0>,
    G3Lhs: BivectorProduct<Vector<T>, Output = G2_1>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = G2_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Vector<T>) -> Self::Output {
        self.1.bivector_prod(rhs) + self.3.bivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2_0, G2_1, G2_2> BivectorProduct<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: BivectorProduct<Bivector<T>, Output = G2_0>,
    G2Lhs: BivectorProduct<Bivector<T>, Output = G2_1>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G2_0: std::ops::Add<G2_1, Output = G2_2>,
{
    type Output = G2_2;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        self.0.bivector_prod(rhs) + self.2.bivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G2_0> BivectorProduct<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: BivectorProduct<Trivector<T>, Output = G2_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G2_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        self.1.bivector_prod(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G2_0,
        G2_1,
        G2_2,
        G2_3,
        G2_4,
        G2_5,
        G2_6,
        G2_7,
        G2_8,
        G2_9,
        G2_10,
    > BivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: BivectorProduct<G2Rhs, Output = G2_0>,
    G1Lhs: BivectorProduct<G1Rhs, Output = G2_1>,
    G1Lhs: BivectorProduct<G3Rhs, Output = G2_2>,
    G2Lhs: BivectorProduct<G0Rhs, Output = G2_3>,
    G2Lhs: BivectorProduct<G2Rhs, Output = G2_4>,
    G3Lhs: BivectorProduct<G1Rhs, Output = G2_5>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G2_0: std::ops::Add<G2_1, Output = G2_6>,
    G2_6: std::ops::Add<G2_2, Output = G2_7>,
    G2_7: std::ops::Add<G2_3, Output = G2_8>,
    G2_8: std::ops::Add<G2_4, Output = G2_9>,
    G2_9: std::ops::Add<G2_5, Output = G2_10>,
{
    type Output = G2_10;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn bivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.0.bivector_prod(rhs.2)
            + self.1.bivector_prod(rhs.1)
            + self.1.bivector_prod(rhs.3)
            + self.2.bivector_prod(rhs.0)
            + self.2.bivector_prod(rhs.2)
            + self.3.bivector_prod(rhs.1)
    }
}
impl TrivectorProduct<Zero> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<T> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Vector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Bivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Trivector<T>> for Zero
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for Zero
where
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Zero> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Zero> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<f32> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: f32) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<f64> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: f64) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Vector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<f32>) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Vector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<f64>) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Bivector<f32>> for f32 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<f32>) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Bivector<f64>> for f64 {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<f64>) -> Self::Output {
        Zero
    }
}
impl TrivectorProduct<Trivector<f32>> for f32 {
    type Output = Trivector<f32>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<f32>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl TrivectorProduct<Trivector<f64>> for f64 {
    type Output = Trivector<f64>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<f64>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self, rhs.xyz),
            }
        }
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f32
where
    f32: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.3)
    }
}
impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for f64
where
    f64: TrivectorProduct<G3Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.3)
    }
}
impl<T> TrivectorProduct<Zero> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<T> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Vector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Bivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(
                    std::intrinsics::fadd_fast(
                        std::intrinsics::fmul_fast(self.x, rhs.yz),
                        -std::intrinsics::fmul_fast(self.y, rhs.xz),
                    ),
                    std::intrinsics::fmul_fast(self.z, rhs.xy),
                ),
            }
        }
    }
}
impl<T> TrivectorProduct<Trivector<T>> for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Vector<T>
where
    T: num_traits::Float,
    Vector<T>: TrivectorProduct<G2Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.2)
    }
}
impl<T> TrivectorProduct<Zero> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<T> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: T) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Vector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<T>) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fadd_fast(
                    std::intrinsics::fadd_fast(
                        std::intrinsics::fmul_fast(self.xy, rhs.z),
                        -std::intrinsics::fmul_fast(self.xz, rhs.y),
                    ),
                    std::intrinsics::fmul_fast(self.yz, rhs.x),
                ),
            }
        }
    }
}
impl<T> TrivectorProduct<Bivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Trivector<T>> for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Bivector<T>
where
    T: num_traits::Float,
    Bivector<T>: TrivectorProduct<G1Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.1)
    }
}
impl<T> TrivectorProduct<Zero> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<T> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: T) -> Self::Output {
        unsafe {
            Trivector {
                xyz: std::intrinsics::fmul_fast(self.xyz, rhs),
            }
        }
    }
}
impl<T> TrivectorProduct<Vector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Bivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        Zero
    }
}
impl<T> TrivectorProduct<Trivector<T>> for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        Zero
    }
}
impl<T, G0Rhs, G1Rhs, G2Rhs, G3Rhs, G3_0> TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Trivector<T>
where
    T: num_traits::Float,
    Trivector<T>: TrivectorProduct<G0Rhs, Output = G3_0>,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.trivector_prod(rhs.0)
    }
}
impl<G0Lhs, G1Lhs, G2Lhs, G3Lhs> TrivectorProduct<Zero> for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = Zero;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Zero) -> Self::Output {
        Zero
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3_0> TrivectorProduct<T>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G3Lhs: TrivectorProduct<T, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: T) -> Self::Output {
        self.3.trivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3_0> TrivectorProduct<Vector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G2Lhs: TrivectorProduct<Vector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Vector<T>) -> Self::Output {
        self.2.trivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3_0> TrivectorProduct<Bivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G1Lhs: TrivectorProduct<Bivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Bivector<T>) -> Self::Output {
        self.1.trivector_prod(rhs)
    }
}
impl<T, G0Lhs, G1Lhs, G2Lhs, G3Lhs, G3_0> TrivectorProduct<Trivector<T>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    T: num_traits::Float,
    G0Lhs: TrivectorProduct<Trivector<T>, Output = G3_0>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
{
    type Output = G3_0;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Trivector<T>) -> Self::Output {
        self.0.trivector_prod(rhs)
    }
}
impl<
        G0Lhs,
        G1Lhs,
        G2Lhs,
        G3Lhs,
        G0Rhs,
        G1Rhs,
        G2Rhs,
        G3Rhs,
        G3_0,
        G3_1,
        G3_2,
        G3_3,
        G3_4,
        G3_5,
        G3_6,
    > TrivectorProduct<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>>
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: TrivectorProduct<G3Rhs, Output = G3_0>,
    G1Lhs: TrivectorProduct<G2Rhs, Output = G3_1>,
    G2Lhs: TrivectorProduct<G1Rhs, Output = G3_2>,
    G3Lhs: TrivectorProduct<G0Rhs, Output = G3_3>,
    G0Lhs: Copy,
    G1Lhs: Copy,
    G2Lhs: Copy,
    G3Lhs: Copy,
    G0Rhs: Copy,
    G1Rhs: Copy,
    G2Rhs: Copy,
    G3Rhs: Copy,
    G3_0: std::ops::Add<G3_1, Output = G3_4>,
    G3_4: std::ops::Add<G3_2, Output = G3_5>,
    G3_5: std::ops::Add<G3_3, Output = G3_6>,
{
    type Output = G3_6;
    #[inline]
    #[allow(unused_variables, unused_unsafe)]
    fn trivector_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
        self.0.trivector_prod(rhs.3)
            + self.1.trivector_prod(rhs.2)
            + self.2.trivector_prod(rhs.1)
            + self.3.trivector_prod(rhs.0)
    }
}
impl std::ops::Neg for Zero {
    type Output = Zero;
    #[inline]
    fn neg(self) -> Self::Output {
        Zero
    }
}
impl<T> std::ops::Neg for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl<T> std::ops::Neg for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        Bivector {
            xy: -self.xy,
            xz: -self.xz,
            yz: -self.yz,
        }
    }
}
impl<T> std::ops::Neg for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        Trivector { xyz: -self.xyz }
    }
}
impl<G0, G1, G2, G3, G0Out, G1Out, G2Out, G3Out> std::ops::Neg for Multivector<G0, G1, G2, G3>
where
    G0: std::ops::Neg<Output = G0Out>,
    G1: std::ops::Neg<Output = G1Out>,
    G2: std::ops::Neg<Output = G2Out>,
    G3: std::ops::Neg<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    fn neg(self) -> Self::Output {
        Multivector(
            std::ops::Neg::neg(self.0),
            std::ops::Neg::neg(self.1),
            std::ops::Neg::neg(self.2),
            std::ops::Neg::neg(self.3),
        )
    }
}
impl Reverse for Zero {
    type Output = Zero;
    #[inline]
    fn rev(self) -> Self::Output {
        Zero
    }
}
impl<T> Reverse for T
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    fn rev(self) -> Self::Output {
        self
    }
}
impl<T> Reverse for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    fn rev(self) -> Self::Output {
        Vector {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}
impl<T> Reverse for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    fn rev(self) -> Self::Output {
        Bivector {
            xy: -self.xy,
            xz: -self.xz,
            yz: -self.yz,
        }
    }
}
impl<T> Reverse for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    fn rev(self) -> Self::Output {
        Trivector { xyz: -self.xyz }
    }
}
impl<G0, G1, G2, G3, G0Out, G1Out, G2Out, G3Out> Reverse for Multivector<G0, G1, G2, G3>
where
    G0: Reverse<Output = G0Out>,
    G1: Reverse<Output = G1Out>,
    G2: Reverse<Output = G2Out>,
    G3: Reverse<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    fn rev(self) -> Self::Output {
        Multivector(
            Reverse::rev(self.0),
            Reverse::rev(self.1),
            Reverse::rev(self.2),
            Reverse::rev(self.3),
        )
    }
}
impl Dual for Zero {
    type Output = Zero;
    #[inline]
    fn dual(self) -> Self::Output {
        Zero
    }
}
impl<T> Dual for T
where
    T: num_traits::Float,
{
    type Output = Trivector<T>;
    #[inline]
    fn dual(self) -> Self::Output {
        Trivector { xyz: self }
    }
}
impl<T> Dual for Vector<T>
where
    T: num_traits::Float,
{
    type Output = Bivector<T>;
    #[inline]
    fn dual(self) -> Self::Output {
        Bivector {
            xy: self.z,
            xz: -self.y,
            yz: self.x,
        }
    }
}
impl<T> Dual for Bivector<T>
where
    T: num_traits::Float,
{
    type Output = Vector<T>;
    #[inline]
    fn dual(self) -> Self::Output {
        Vector {
            x: self.yz,
            y: -self.xz,
            z: self.xy,
        }
    }
}
impl<T> Dual for Trivector<T>
where
    T: num_traits::Float,
{
    type Output = T;
    #[inline]
    fn dual(self) -> Self::Output {
        self.xyz
    }
}
impl<G0, G1, G2, G3, G0Out, G1Out, G2Out, G3Out> Dual for Multivector<G0, G1, G2, G3>
where
    G0: Dual<Output = G3Out>,
    G1: Dual<Output = G2Out>,
    G2: Dual<Output = G1Out>,
    G3: Dual<Output = G0Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    #[inline]
    fn dual(self) -> Self::Output {
        Multivector(
            Dual::dual(self.3),
            Dual::dual(self.2),
            Dual::dual(self.1),
            Dual::dual(self.0),
        )
    }
}
impl ToF32 for Zero {
    type Output = Zero;
    fn to_f32(self) -> Self::Output {
        Zero
    }
}
impl ToF64 for Zero {
    type Output = Zero;
    fn to_f64(self) -> Self::Output {
        Zero
    }
}
impl ToF32 for f64 {
    type Output = f32;
    fn to_f32(self) -> Self::Output {
        self as f32
    }
}
impl ToF64 for f32 {
    type Output = f64;
    fn to_f64(self) -> Self::Output {
        self as f64
    }
}
impl ToF32 for Vector<f64> {
    type Output = Vector<f32>;
    fn to_f32(self) -> Self::Output {
        Vector {
            x: self.x as f32,
            y: self.y as f32,
            z: self.z as f32,
        }
    }
}
impl ToF64 for Vector<f32> {
    type Output = Vector<f64>;
    fn to_f64(self) -> Self::Output {
        Vector {
            x: self.x as f64,
            y: self.y as f64,
            z: self.z as f64,
        }
    }
}
impl ToF32 for Bivector<f64> {
    type Output = Bivector<f32>;
    fn to_f32(self) -> Self::Output {
        Bivector {
            xy: self.xy as f32,
            xz: self.xz as f32,
            yz: self.yz as f32,
        }
    }
}
impl ToF64 for Bivector<f32> {
    type Output = Bivector<f64>;
    fn to_f64(self) -> Self::Output {
        Bivector {
            xy: self.xy as f64,
            xz: self.xz as f64,
            yz: self.yz as f64,
        }
    }
}
impl ToF32 for Trivector<f64> {
    type Output = Trivector<f32>;
    fn to_f32(self) -> Self::Output {
        Trivector {
            xyz: self.xyz as f32,
        }
    }
}
impl ToF64 for Trivector<f32> {
    type Output = Trivector<f64>;
    fn to_f64(self) -> Self::Output {
        Trivector {
            xyz: self.xyz as f64,
        }
    }
}
impl<G0Lhs, G0Out, G1Lhs, G1Out, G2Lhs, G2Out, G3Lhs, G3Out> ToF32
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ToF32<Output = G0Out>,
    G1Lhs: ToF32<Output = G1Out>,
    G2Lhs: ToF32<Output = G2Out>,
    G3Lhs: ToF32<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    fn to_f32(self) -> Self::Output {
        Multivector(
            self.0.to_f32(),
            self.1.to_f32(),
            self.2.to_f32(),
            self.3.to_f32(),
        )
    }
}
impl<G0Lhs, G0Out, G1Lhs, G1Out, G2Lhs, G2Out, G3Lhs, G3Out> ToF64
    for Multivector<G0Lhs, G1Lhs, G2Lhs, G3Lhs>
where
    G0Lhs: ToF64<Output = G0Out>,
    G1Lhs: ToF64<Output = G1Out>,
    G2Lhs: ToF64<Output = G2Out>,
    G3Lhs: ToF64<Output = G3Out>,
{
    type Output = Multivector<G0Out, G1Out, G2Out, G3Out>;
    fn to_f64(self) -> Self::Output {
        Multivector(
            self.0.to_f64(),
            self.1.to_f64(),
            self.2.to_f64(),
            self.3.to_f64(),
        )
    }
}
