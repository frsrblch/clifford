pub use geo_traits::*;
pub use num_traits::*;

macros::algebra! {
    x ^ 2 == 1,
    y ^ 2 == 1,
    z ^ 2 == 1,
}

impl<T> Bivector<T>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.unit().sqrt()
    }
}

impl<T> Unit<Bivector<T>>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.value() + Scalar::<T>::one()
    }
}

impl<T> Motor<T>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.unit().sqrt()
    }
}

impl<T> Unit<Motor<T>>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.value() + Scalar::<T>::one()
    }
}

impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    T: Float + FloatConst,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
        let bivector = rng.gen::<Unit<Bivector<T>>>();
        let angle = Scalar::new(rng.gen::<T>() * T::PI());
        let (sin, cos) = angle.sin_cos();
        Unit::assert(bivector.value() * cos + sin)
    }
}

impl<T> Motor<T>
where
    T: Float,
{
    #[inline]
    pub fn from_plane_and_angle(plane: Unit<Bivector<T>>, angle: Scalar<T>) -> Motor<T> {
        let neg_half = Scalar::new(-T::one() / (T::one() + T::one()));
        let (sin, cos) = Float::sin_cos(angle * neg_half);
        plane.value() * sin + cos
    }

    #[inline]
    pub fn from_axis_and_angle(axis: Unit<Vector<T>>, angle: Scalar<T>) -> Motor<T> {
        let plane = Unit::assert(axis.value().dual());
        Self::from_plane_and_angle(plane, angle)
    }

    #[inline]
    pub fn angle(self) -> Scalar<T> {
        let two = T::one() + T::one();
        self.scalar().acos() * Scalar { s: two }
    }

    #[inline]
    pub fn plane(self) -> Unit<Bivector<T>> {
        (-self.bivector()).unit()
    }
}
