proc_macros::clifford!(3, 0, 1);

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

pub const fn point(x: f64, y: f64, z: f64) -> Vector {
    Vector::new(x, y, z, 1.)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            e14: 1.,
            ..Default::default()
        };

        let e24 = Bivector {
            e24: 1.,
            ..Default::default()
        };

        let expected = Even {
            e34: -1.,
            ..Default::default()
        };

        assert_eq!(expected, e14.antigeo(e24));
    }

    #[test]
    fn bivector_bulk_weight() {
        let b = Bivector::new(1., 1., 1., 1., 1., 1.);

        let bulk = Bivector {
            e12: 1.,
            e23: 1.,
            e13: 1.,
            ..Default::default()
        };
        let weight = Bivector {
            e14: 1.,
            e24: 1.,
            e34: 1.,
            ..Default::default()
        };

        assert_eq!(bulk, b.bulk());
        assert_eq!(weight, b.weight());
    }

    #[test]
    fn ideal_point() {
        let real = point(1., 2., 3.);
        let ideal = Vector::new(1., 2., 3., 0.);

        assert!(!real.is_ideal());
        assert!(ideal.is_ideal());
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
