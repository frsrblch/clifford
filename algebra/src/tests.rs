use super::*;
use Square::*;

pub fn ga_2d() -> Algebra {
    let x = Pos.basis('x');
    let y = Pos.basis('y');
    Algebra::new([x, y])
}

pub fn ga_3d() -> Algebra {
    let x = Pos.basis('x');
    let y = Pos.basis('y');
    let z = Pos.basis('z');
    let bases = [x, y, z];
    let fields = [
        format_ident!("xy"),
        format_ident!("yz"),
        format_ident!("zx"),
        format_ident!("xyz"),
    ];
    Algebra::new_with_fields(bases, fields)
}

pub fn pga_3d() -> Algebra {
    let x = Pos.basis('x');
    let y = Pos.basis('y');
    let z = Pos.basis('z');
    let w = Zero.basis('w');
    let bases = [x, y, z, w];
    let fields = [
        format_ident!("xy"),
        format_ident!("yz"),
        format_ident!("zx"),
        format_ident!("wz"),
        format_ident!("wy"),
        format_ident!("wx"),
        format_ident!("zyx"),
        format_ident!("wxy"),
        format_ident!("wzx"),
        format_ident!("wyz"),
    ];
    Algebra::new_with_fields(bases, fields)
}

pub fn cga_3d() -> Algebra {
    let x = Pos.basis('x');
    let y = Pos.basis('y');
    let z = Pos.basis('z');
    let e_pos = Pos.basis('e');
    let e_neg = Neg.basis('E');
    Algebra::new([x, y, z, e_pos, e_neg])
}

#[test]
fn algebra_geo() {
    let algebra = ga_2d();

    let s = Blade::scalar();
    let z = Blade::zero();

    assert_eq!(s, algebra.geo(s, s));
    assert_eq!(z, algebra.geo(z, s));
    assert_eq!(z, algebra.geo(s, z));
    assert_eq!(z, algebra.geo(z, z));
}

#[test]
fn algebra_dot() {
    let algebra = ga_2d();

    let s = Blade::scalar();
    let e1 = Blade(0b1);
    let e2 = Blade(0b10);
    let e12 = Blade(0b11);
    let z = Blade::zero();

    assert_eq!(e1, algebra.dot(s, e1));
    assert_eq!(e1, algebra.dot(e1, s));
    assert_eq!(s, algebra.dot(e1, e1));
    assert_eq!(z, algebra.dot(z, e1));
    assert_eq!(z, algebra.dot(e1, z));
    assert_eq!(z, algebra.dot(z, z));
    assert_eq!(z, algebra.dot(e1, e2));
    assert_eq!(e1, algebra.dot(e12, e2));
    assert_eq!(-e1, algebra.dot(e2, e12));
    assert_eq!(-s, algebra.dot(e12, e12));
}

#[test]
fn algebra_wedge() {
    let algebra = ga_2d();

    let s = Blade::scalar();
    let e1 = Blade(0b1);
    let e2 = Blade(0b10);
    let e12 = Blade(0b11);
    let z = Blade::zero();

    assert_eq!(e1, algebra.wedge(s, e1));
    assert_eq!(e1, algebra.wedge(e1, s));
    assert_eq!(z, algebra.wedge(e1, e1));
    assert_eq!(e12, algebra.wedge(e1, e2));
    assert_eq!(-e12, algebra.wedge(e2, e1));
    assert_eq!(z, algebra.wedge(z, e1));
    assert_eq!(z, algebra.wedge(e1, z));
    assert_eq!(z, algebra.wedge(z, z));
}

#[test]
fn algebra_left_comp() {
    let algebra = ga_2d();

    let z = Blade::zero();
    let s = Blade::scalar();
    let e1 = Blade(0b1);
    let e2 = Blade(0b10);
    let e12 = Blade(0b11);

    assert_eq!(z, algebra.left_comp(z));
    assert_eq!(s, algebra.left_comp(e12));
    assert_eq!(-e2, algebra.left_comp(e1));
    assert_eq!(e1, algebra.left_comp(e2));
    assert_eq!(e12, algebra.left_comp(s));
}

#[test]
fn algebra_right_comp() {
    let algebra = ga_2d();

    let z = Blade::zero();
    let s = Blade::scalar();
    let e1 = Blade(0b1);
    let e2 = Blade(0b10);
    let e12 = Blade(0b11);

    assert_eq!(z, algebra.right_comp(z));
    assert_eq!(s, algebra.right_comp(e12));
    assert_eq!(e2, algebra.right_comp(e1));
    assert_eq!(-e1, algebra.right_comp(e2));
    assert_eq!(e12, algebra.right_comp(s));

    let algebra = ga_3d();
    let e3 = Blade(0b100);
    let e23 = Blade(0b110);
    let e31 = Blade(0b101);

    assert_eq!(e23, algebra.right_comp(e1));
    assert_eq!(e31, algebra.right_comp(e2));
    assert_eq!(e12, algebra.right_comp(e3));
}

#[test]
fn algebra_geo_zero() {
    let algebra = pga_3d();

    let w = Blade(0b1000);
    let xw = Blade(0b1001);
    let zero = Blade::zero();

    assert_eq!(zero, algebra.geo(w, w));
    assert_eq!(zero, algebra.geo(w, xw));
    assert_eq!(zero, algebra.geo(xw, w));
    assert_eq!(zero, algebra.geo(xw, xw));
}

#[test]
#[allow(non_snake_case)]
fn antidot() {
    let a = pga_3d();

    let s = Blade(0);
    let I = Blade(0b1111);
    let e1 = Blade(0b0001);
    let e4 = Blade(0b1000);
    let e124 = Blade(0b1011);
    let e123 = Blade(0b0111);

    assert_eq!(-e123, a.antidot(e4, s));
    assert_eq!(-I, a.antidot(e4, e4));
    assert_eq!(I, a.antidot(e124, e124));
    assert_eq!(I, a.antidot(I, I));
    assert_eq!(Blade::zero(), a.antidot(e1, e1));
}

#[test]
#[allow(non_snake_case)]
fn antiwedge() {
    let a = pga_3d();

    let z = Blade::zero();
    let s = Blade(0);
    let e1 = Blade(0b0001);
    let e4 = Blade(0b1000);
    let e41 = Blade(0b1001);
    let e23 = Blade(0b0110);
    let e423 = Blade(0b1110);

    assert_eq!(-s, a.antiwedge(e41, e23));
    assert_eq!(-s, a.antiwedge(e423, e1));
    assert_eq!(s, a.antiwedge(e1, e423));
    assert_eq!(z, a.antiwedge(e1, e1));
    assert_eq!(z, a.antiwedge(e1, e4));
}

#[test]
fn algebra_sym_comp() {
    assert!(!ga_2d().symmetric_complements());
    assert!(ga_3d().symmetric_complements());
    assert!(!pga_3d().symmetric_complements());
    assert!(cga_3d().symmetric_complements());
}

#[test]
fn ga_3d_blade_fields() {
    let a = crate::tests::ga_3d();
    assert_eq!("x", a.fields[Blade(0b1)].to_string());
    assert_eq!("y", a.fields[Blade(0b10)].to_string());
    assert_eq!("z", a.fields[Blade(0b100)].to_string());
    assert_eq!("xy", a.fields[Blade(0b11)].to_string());
    assert_eq!("yz", a.fields[Blade(0b110)].to_string());
    assert_eq!("zx", a.fields[Blade(0b101)].to_string());
    assert_eq!("xyz", a.fields[Blade(0b111)].to_string());
}

#[test]
fn pga_3d_blade_fields() {
    let a = crate::tests::pga_3d();
    assert_eq!("x", a.fields[Blade(0b0001)].to_string());
    assert_eq!("y", a.fields[Blade(0b0010)].to_string());
    assert_eq!("z", a.fields[Blade(0b0100)].to_string());
    assert_eq!("w", a.fields[Blade(0b1000)].to_string());
    assert_eq!("xy", a.fields[Blade(0b0011)].to_string());
    assert_eq!("yz", a.fields[Blade(0b0110)].to_string());
    assert_eq!("zx", a.fields[Blade(0b0101)].to_string());
    assert_eq!("wx", a.fields[Blade(0b1001)].to_string());
    assert_eq!("wy", a.fields[Blade(0b1010)].to_string());
    assert_eq!("wz", a.fields[Blade(0b1100)].to_string());
    assert_eq!("zyx", a.fields[Blade(0b0111)].to_string());
    assert_eq!("wxy", a.fields[Blade(0b1011)].to_string());
    assert_eq!("wzx", a.fields[Blade(0b1101)].to_string());
    assert_eq!("wyz", a.fields[Blade(0b1110)].to_string());
}

#[test]
fn type_grades_iter() {
    let a = pga_3d();
    assert_eq!(
        vec![0],
        TypeGrades::new(&a, Type::Grade(0)).collect::<Vec<_>>()
    );
    assert_eq!(
        vec![1],
        TypeGrades::new(&a, Type::Grade(1)).collect::<Vec<_>>()
    );
    assert_eq!(
        vec![0, 2, 4],
        TypeGrades::new(&a, Type::Motor).collect::<Vec<_>>()
    );
    assert_eq!(
        vec![1, 3],
        TypeGrades::new(&a, Type::Flector).collect::<Vec<_>>()
    );
}

#[test]
fn type_blades_sorted_iter() {
    let a = pga_3d();
    let s = Blade(0b0000);
    let xy = Blade(0b0011);
    let yz = Blade(0b0110);
    let zx = Blade(0b0101);
    let wx = Blade(0b1001);
    let wy = Blade(0b1010);
    let wz = Blade(0b1100);
    let xyzw = Blade(0b1111);
    assert_eq!(
        vec![s, xy, zx, yz, wx, wy, wz, xyzw],
        SortedTypeBlades::new(&a, Type::Motor).collect::<Vec<_>>()
    );
}

#[test]
fn duals_even_dimension() {
    let algebra = pga_3d();
    let n = algebra.dim();
    for blade in Blades::from(&algebra) {
        let k = blade.grade();
        let sign = if (k * (n - 1)) & 1 == 1 {
            -Blade(0)
        } else {
            Blade(0)
        };
        assert_eq!(algebra.left_comp(blade), algebra.right_comp(blade) ^ sign);
    }
}

#[test]
fn duals_odd_dimension() {
    let algebra = ga_3d();
    let n = algebra.dim();
    for blade in Blades::from(&algebra) {
        let k = blade.grade();
        let sign = if (k * (n - 1)) & 1 == 1 {
            -Blade(0)
        } else {
            Blade(0)
        };
        assert_eq!(algebra.left_comp(blade), algebra.right_comp(blade) ^ sign);
    }
}

#[test]
fn blade_products() {
    let a = ga_3d();
    let x = Blade(0b001);
    let y = Blade(0b010);
    let z = Blade(0b100);
    let xy = Blade(0b011);
    let yz = Blade(0b110);
    let zx = Blade(0b101);
    let xyz = Blade(0b111);

    assert_eq!(xy, a.geo(x, y));
    assert_eq!(-xy, a.geo(y, x));
    assert_eq!(yz, a.geo(y, z));
    assert_eq!(-yz, a.geo(z, y));
    assert_eq!(zx, a.geo(z, x));
    assert_eq!(-zx, a.geo(x, z));
    assert_eq!(xyz, a.geo(x, yz));
    assert_eq!(xyz, a.geo(yz, x));
    assert_eq!(xyz, a.geo(xy, z));
    assert_eq!(xyz, a.geo(z, xy));
    assert_eq!(xyz, a.geo(y, zx));
    assert_eq!(xyz, a.geo(zx, y));
}

#[test]
fn manual_blade_ordering() {
    use crate::Square::*;
    let algebra = Algebra::new_with_fields([Pos.basis('x'), Pos.basis('y')], [format_ident!("yx")]);

    assert!(!algebra.ordering.is_flipped(Blade(0)));
    assert!(!algebra.ordering.is_flipped(Blade(1)));
    assert!(!algebra.ordering.is_flipped(Blade(2)));
    assert!(algebra.ordering.is_flipped(Blade(3)));
}

#[test]
fn left_contraction() {
    let a = ga_2d();
    let x = Blade(0b1);
    let xy = Blade(0b11);

    assert_eq!(Blade(0b10), a.left_con(x, xy));
    assert_eq!(Blade::zero(), a.left_con(xy, x));
    assert_eq!(Blade(0), a.left_con(x, x));
}

#[test]
fn right_contraction() {
    let a = ga_2d();
    let x = Blade(0b1);
    let xy = Blade(0b11);

    assert_eq!(Blade::zero(), a.right_con(x, xy));
    assert_eq!(-Blade(0b10), a.right_con(xy, x));
    assert_eq!(Blade(0), a.right_con(x, x));
}
