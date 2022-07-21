proc_macros::clifford!(3, 0, 1, [x, y, z, w]);

pub type Motor<T> = Multivector<T, Zero, Bivector<T>, Zero, Quadvector<T>>;
pub type Rotor<T> = Multivector<Zero, Zero, Bivector<T>, Zero, Quadvector<T>>;

pub trait Join<Rhs> {
    type Output;
    fn join(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> Join<Rhs> for Lhs
where
    Lhs: Wedge<Rhs>,
{
    type Output = Lhs::Output;

    fn join(self, rhs: Rhs) -> Self::Output {
        self.wedge(rhs)
    }
}

pub trait Meet<Rhs> {
    type Output;
    fn meet(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> Meet<Rhs> for Lhs
where
    Lhs: Antiwedge<Rhs>,
{
    type Output = Lhs::Output;

    fn meet(self, rhs: Rhs) -> Self::Output {
        self.antiwedge(rhs)
    }
}

pub trait Project<Target> {
    type Output;
    fn project(self, target: Target) -> Self::Output;
}

impl<T, Target, TargetWeight, TargetComp, Wedged> Project<Target> for T
where
    Target: Weight<Output = TargetWeight> + Copy,
    TargetWeight: LeftComplement<Output = TargetComp>,
    TargetComp: Wedge<T, Output = Wedged>,
    Wedged: Antiwedge<Target>,
{
    type Output = Wedged::Output;

    #[inline]
    fn project(self, target: Target) -> Self::Output {
        target.weight().left_comp().wedge(self).antiwedge(target)
    }
}

pub trait Antiproject<Target> {
    type Output;
    fn antiproject(self, target: Target) -> Self::Output;
}

impl<T, Target, TargetWeight, TargetComp, Antiwedged> Antiproject<Target> for T
where
    Target: Weight<Output = TargetWeight> + Copy,
    TargetWeight: LeftComplement<Output = TargetComp>,
    TargetComp: Antiwedge<T, Output = Antiwedged>,
    Antiwedged: Wedge<Target>,
{
    type Output = Antiwedged::Output;

    #[inline]
    fn antiproject(self, target: Target) -> Self::Output {
        target.weight().left_comp().antiwedge(self).wedge(target)
    }
}

#[inline]
pub fn point<T: num_traits::One>(x: T, y: T, z: T) -> Vector<T> {
    Vector::new(x, y, z, T::one())
}

#[inline]
pub fn direction<T: num_traits::Float>(x: T, y: T, z: T) -> Vector<T> {
    Vector {
        x,
        y,
        z,
        w: T::zero(),
    }
    .unit()
}

#[inline]
pub fn translator<T: num_traits::Float>(v: Vector<T>) -> Rotor<T> {
    let half = T::one() / (T::one() + T::one());
    Bivector {
        xy: v.z * half,
        yz: v.x * half,
        xz: -v.y * half,
        xw: T::zero(),
        yw: T::zero(),
        zw: T::zero(),
    } + Quadvector::new(T::one())
}

#[inline]
pub fn rotate_origin<T: num_traits::Float + std::fmt::Debug>(
    from_dir: Vector<T>,
    to_dir: Vector<T>,
) -> Rotor<T> {
    to_dir.geo(from_dir).left_comp().sqrt()
}

impl<T: num_traits::Float> Rotor<T> {
    #[inline]
    pub fn sqrt(self) -> Self {
        let i = Quadvector::new(T::one());
        let two = T::one() + T::one();

        let num: Rotor<T> = self + i;
        let den: T = (two + two * self.4.xyzw).sqrt();

        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_point_motors() {
        let x = direction(1f64, 0., 0.);
        let xy = direction(-1., 1., 0.);
        let motor = rotate_origin(x, xy);

        let pt = point(1., 0., 0.);

        let expected = point(-1. / 2f64.sqrt(), 1. / 2f64.sqrt(), 0.);
        let actual = motor.antigeo(pt).antigeo(motor.antirev()).1;

        assert!(0.0000001 > (expected.x - actual.x).abs());
        assert!(0.0000001 > (expected.y - actual.y).abs());
        assert!(0.0000001 > (expected.z - actual.z).abs());
        assert!(0.0000001 > (expected.w - actual.w).abs());
    }

    #[test]
    fn ideal_line_complement_is_origin_line() {
        let o = point(0., 0., 0.);
        let x = direction(1., 0., 0.);
        let y = direction(0., 1., 0.);
        let z1 = point(0., 0., 1.);
        assert_eq!(o.wedge(z1), y.wedge(x).left_comp());
    }

    #[test]
    fn translation_test() {
        let translator = translator(Vector::new(2., 3., 5., 0.));
        let one = point(1., 2., 3.);
        let expected = point(3., 5., 8.);
        let actual = translator.antigeo(one).antigeo(translator.antirev()).1;
        assert_eq!(expected, actual);
    }

    #[test]
    fn vec_mul() {
        let a = point(2., 3., 4.);
        let b = point(3., 5., 7.);
        let c = point(0., 0., 0.);

        let line = a.wedge(b);
        let plane = line.wedge(c);

        dbg!(line, plane, a * b);

        // panic!("done");
    }

    #[test]
    fn plane_dual() {
        let o = point(0., 0., 0.);
        let x = point(1., 0., 0.);
        let y = point(0., 1., 0.);

        let plane = o.wedge(x).wedge(y);
        dbg!(plane);

        // panic!();
    }

    #[test]
    fn antigeometric_product() {
        let e14 = Bivector {
            xw: 1.,
            ..Default::default()
        };

        let e24 = Bivector {
            yw: 1.,
            ..Default::default()
        };

        let expected = Bivector {
            zw: -1.,
            ..Default::default()
        } + 0.0
            + Quadvector::default();

        assert_eq!(expected, e14.antigeo(e24));
    }

    #[test]
    fn bivector_bulk_weight() {
        let b = Bivector::new(1., 1., 1., 1., 1., 1.);

        let bulk = Bivector {
            xy: 1.,
            yz: 1.,
            xz: 1.,
            ..Default::default()
        };
        let weight = Bivector {
            xw: 1.,
            yw: 1.,
            zw: 1.,
            ..Default::default()
        };

        assert_eq!(bulk, b.bulk());
        assert_eq!(weight, b.weight());
    }

    #[test]
    fn project_point_onto_plane() {
        let p = point(2., 3., 5.);

        let a = point(0., 0., 0.);
        let b = point(1., 0., 0.);
        let c = point(0., 1., 0.);
        let xy_plane = a.wedge(b).wedge(c);

        let expected = point(2., 3., 0.);

        assert_eq!(expected, p.project(xy_plane));
    }
}
