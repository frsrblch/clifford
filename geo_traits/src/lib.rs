pub trait Geo<Rhs> {
    type Output;
    fn geo(self, rhs: Rhs) -> Self::Output;
}

pub trait Wedge<Rhs> {
    type Output;
    fn wedge(self, rhs: Rhs) -> Self::Output;
}

pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

pub trait Antigeo<Rhs> {
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

pub trait Commutator<Rhs> {
    type Output;
    fn com(self, rhs: Rhs) -> Self::Output;
}

pub trait Dual {
    type Output;
    fn dual(self) -> Self::Output;
}

pub trait LeftComplement {
    type Output;
    fn left_comp(self) -> Self::Output;
}

pub trait RightComplement {
    type Output;
    fn right_comp(self) -> Self::Output;
}

pub trait Reverse {
    type Output;
    fn rev(self) -> Self::Output;
}

pub trait Antireverse {
    fn antirev(self) -> Self;
}

pub trait GradeProduct<Lhs, Rhs> {
    type Output;
    fn product(lhs: Lhs, rhs: Rhs) -> Self::Output;
}

pub trait GradeAntiproduct<Lhs, Rhs> {
    fn antiproduct(lhs: Lhs, rhs: Rhs) -> Self;
}

pub trait Norm2 {
    type Output;
    fn norm2(self) -> Self::Output;
}

pub trait Norm {
    type Output;
    fn norm(self) -> Self::Output;
}

pub trait Sandwich<Rhs> {
    type Output;
    fn sandwich(self, rhs: Rhs) -> Self::Output;
}

pub trait Antisandwich<Rhs> {
    type Output;
    fn antisandwich(self, rhs: Rhs) -> Self::Output;
}

pub trait Inverse {
    type Output;
    fn inv(self) -> Self::Output;
}

pub trait Unitize {
    type Output;
    fn unit(self) -> Self::Output;
}

pub trait FloatType {
    type Float: num_traits::Float;
}
