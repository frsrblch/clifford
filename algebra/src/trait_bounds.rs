use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use strum::EnumIter;

use crate::{Algebra, BinaryTrait, Float, OverType, UnaryTrait, Value};
use std::collections::{BTreeMap, BTreeSet};

pub trait Insert<Item> {
    fn insert(&mut self, item: Item);
}

impl<T, const N: usize> Insert<[T; N]> for TraitBounds
where
    Self: Insert<T>,
{
    fn insert(&mut self, items: [T; N]) {
        IntoIterator::into_iter(items).for_each(|item| self.insert(item));
    }
}

impl Insert<ParameterizedType> for TraitBounds {
    fn insert(&mut self, item: ParameterizedType) {
        self.insert(item.param);
        self.insert(item.mag);
    }
}

impl Insert<TypeParam> for TraitBounds {
    fn insert(&mut self, item: TypeParam) {
        self.bounds.entry(item).or_default();
    }
}

impl Insert<MagParam> for TraitBounds {
    fn insert(&mut self, item: MagParam) {
        self.insert(TypeParam::Mag(item));
    }
}

impl Insert<FloatParam> for TraitBounds {
    fn insert(&mut self, item: FloatParam) {
        self.insert(TypeParam::Float(item));
    }
}

impl Insert<TraitBound> for TraitBounds {
    fn insert(&mut self, item: TraitBound) {
        for t in item.generics() {
            self.insert(t);
        }
        self.bounds.entry(item.ty()).or_default().insert(item);
    }
}

impl Insert<Lifetime> for TraitBounds {
    fn insert(&mut self, item: Lifetime) {
        self.bounds.entry(TypeParam::Lifetime(item)).or_default();
    }
}

#[derive(Debug, Default, Clone)]
pub struct TraitBounds {
    bounds: BTreeMap<TypeParam, BTreeSet<TraitBound>>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TypeParam {
    Lifetime(Lifetime),
    Float(FloatParam),
    Mag(MagParam),
    Type(ParameterizedType),
}

impl From<FloatParam> for TypeParam {
    fn from(value: FloatParam) -> Self {
        TypeParam::Float(value)
    }
}

impl From<Float> for TypeParam {
    fn from(value: Float) -> Self {
        TypeParam::Float(value.into())
    }
}

impl From<MagParam> for TypeParam {
    fn from(value: MagParam) -> Self {
        TypeParam::Mag(value)
    }
}

impl From<ParameterizedType> for TypeParam {
    fn from(value: ParameterizedType) -> Self {
        TypeParam::Type(value)
    }
}

impl ToTokens for TypeParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            TypeParam::Lifetime(lifetime) => lifetime.to_tokens(tokens),
            TypeParam::Float(float) => float.to_tokens(tokens),
            TypeParam::Mag(mag) => mag.to_tokens(tokens),
            TypeParam::Type(ty) => ty.to_tokens(tokens),
        }
    }
}

impl TypeParam {
    pub fn generics(&self) -> impl Iterator<Item = TypeParam> {
        match self {
            TypeParam::Lifetime(lifetime) => [Some(TypeParam::Lifetime(*lifetime)), None],
            TypeParam::Float(float) => match float {
                FloatParam::Float(_) => [None; 2],
                &float => [Some(float.into()), None],
            },
            TypeParam::Mag(mag) => match mag {
                MagParam::Mag(_) => [None; 2],
                &mag => [Some(mag.into()), None],
            },
            TypeParam::Type(ty) => [
                ty.param.is_generic().then_some(ty.param.into()),
                ty.mag.is_generic().then_some(ty.mag.into()),
            ],
        }
        .into_iter()
        .flatten()
    }
}

impl FromIterator<FloatParam> for TraitBounds {
    fn from_iter<T: IntoIterator<Item = FloatParam>>(iter: T) -> Self {
        let mut new = Self::default();
        for ty in iter {
            if ty.is_generic() {
                new.bounds.insert(TypeParam::Float(ty), Default::default());
            }
        }
        new
    }
}

impl TraitBounds {
    pub fn geo_types(
        lhs: OverType,
        rhs: OverType,
        algebra: &Algebra,
        op: BinaryTrait,
    ) -> Option<(Self, [FloatParam; 3], [MagParam; 3])> {
        use FloatParam::{T, U, V};
        use Mag::Any;
        use MagParam::{A, B, C};
        let mut bounds = Self::default();
        match (lhs, rhs) {
            (OverType::Float(lf), OverType::Float(rf)) => {
                if lf == rf {
                    let f = FloatParam::Float(lf);
                    Some((bounds, [f; 3], [Any.into(); 3]))
                } else {
                    None
                }
            }
            (OverType::Float(lf), _) => {
                let f = FloatParam::Float(lf);
                bounds.insert(f.mul(U, U));
                bounds.insert(U.copy());
                bounds.insert(B);
                Some((bounds, [f, U, U], [Any.into(), B, Any.into()]))
            }
            (_, OverType::Float(rf)) => {
                let f = FloatParam::Float(rf);
                bounds.insert(T.mul(f, T));
                bounds.insert(T.copy());
                bounds.insert(A);
                Some((bounds, [T, f, T], [A, Any.into(), Any.into()]))
            }
            _ => {
                let output_can_be_unit = {
                    const N: usize = 10;
                    (0..N).all(|_| {
                        let mut lhs = Value::gen(lhs.into(), algebra);
                        let mut rhs = Value::gen(rhs.into(), algebra);
                        lhs.unit(algebra);
                        rhs.unit(algebra);
                        if let Some(output) = lhs.mul(&rhs, algebra) {
                            output.is_unit(algebra)
                        } else {
                            false
                        }
                    })
                };
                let abc = if output_can_be_unit {
                    if op == BinaryTrait::Div {
                        bounds.insert(A.div(B, C));
                    } else {
                        bounds.insert(A.mul(B, C));
                    }
                    [A, B, C]
                } else {
                    bounds.insert([A, B]);
                    [A, B, Any.into()]
                };
                bounds.insert(T.mul(U, V));
                bounds.insert([T.copy(), U.copy()]);
                Some((bounds, [T, U, V], abc))
            }
        }
    }

    pub fn product_types(
        lhs: OverType,
        rhs: OverType,
    ) -> Option<(Self, [FloatParam; 3], [MagParam; 3])> {
        use FloatParam::{T, U, V};
        use Mag::Any;
        use MagParam::{A, B};
        let mut bounds = Self::default();
        match (lhs, rhs) {
            (OverType::Float(lf), OverType::Float(rf)) => {
                if lf == rf {
                    let f = FloatParam::Float(lf);
                    Some((bounds, [f; 3], [Any.into(); 3]))
                } else {
                    None
                }
            }
            (OverType::Float(lf), _) => {
                let f = FloatParam::Float(lf);
                bounds.insert(f.mul(U, U));
                bounds.insert(B);
                Some((bounds, [f, U, U], [Any.into(), B, Any.into()]))
            }
            (_, OverType::Float(rf)) => {
                let f = FloatParam::Float(rf);
                bounds.insert(T.mul(f, T));
                bounds.insert(A);
                Some((bounds, [T, f, T], [A, Any.into(), Any.into()]))
            }
            _ => {
                let abc = [A, B, Any.into()];
                bounds.insert(abc);
                bounds.insert(T.mul(U, V));
                bounds.insert([T.copy(), U.copy()]);
                Some((bounds, [T, U, V], abc))
            }
        }
    }

    pub fn sum_types(
        lhs: OverType,
        rhs: OverType,
    ) -> Option<(TraitBounds, [FloatParam; 3], [MagParam; 3])> {
        use FloatParam::{T, U, V};
        use Mag::Any;
        use MagParam::{A, B};
        let mut bounds = Self::default();
        match (lhs, rhs) {
            (OverType::Float(f), OverType::Float(_)) => {
                if lhs == rhs {
                    Some((bounds, [f.into(); 3], [Any.into(); 3]))
                } else {
                    None
                }
            }
            (OverType::Type(_), OverType::Type(_)) => {
                if lhs == rhs {
                    bounds.insert(T);
                    bounds.insert(U);
                    bounds.insert(V);
                    bounds.insert(A);
                    bounds.insert(B);
                    Some((bounds, [T, U, V], [A, B, Any.into()]))
                } else {
                    bounds.insert(T);
                    bounds.insert(A);
                    bounds.insert(B);
                    Some((bounds, [T, T, T], [A, B, Any.into()]))
                }
            }
            (OverType::Type(_), OverType::Float(f)) => {
                bounds.insert(T);
                bounds.insert(A);
                Some((bounds, [T, f.into(), T], [A, Any.into(), Any.into()]))
            }
            (OverType::Float(f), OverType::Type(_)) => {
                bounds.insert(U);
                bounds.insert(A);
                Some((bounds, [f.into(), U, U], [Any.into(), A, Any.into()]))
            }
        }
    }

    pub fn with_param(ty: FloatParam) -> Self {
        Self::from_iter([ty])
    }

    pub fn tuv() -> Self {
        let mut new = Self::default();
        new.insert(FloatParam::T);
        new.insert(FloatParam::U);
        new.insert(FloatParam::V);
        new
    }

    pub fn params_and_where_clause(self) -> (TokenStream, TokenStream) {
        let params = {
            let params = self
                .bounds
                .iter()
                .flat_map(|(k, v)| {
                    let k = k.generics();
                    let v = v.iter().flat_map(|v| v.generics());
                    k.chain(v)
                })
                .collect::<BTreeSet<_>>();
            if params.is_empty() {
                quote!()
            } else {
                quote! {
                    < #(#params),* >
                }
            }
        };
        let where_clause = {
            let bounds = self.bounds.iter().fold(quote!(), |mut ts, (ty, bounds)| {
                let bounds = bounds.iter();
                let bounds = quote!(#(#bounds)+*);
                if !bounds.is_empty() {
                    quote!(#ty: #bounds,).to_tokens(&mut ts);
                }
                ts
            });
            if bounds.is_empty() {
                bounds
            } else {
                quote! { where #bounds }
            }
        };
        (params, where_clause)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, EnumIter)]
pub enum Mag {
    Any,
    Unit,
}

impl ToTokens for Mag {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Mag::Any => quote!(Any),
            Mag::Unit => quote!(Unit),
        }
        .to_tokens(tokens);
    }
}

impl std::ops::Mul for Mag {
    type Output = Mag;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Mag::Unit, Mag::Unit) => Mag::Unit,
            _ => Mag::Any,
        }
    }
}

impl std::ops::Div for Mag {
    type Output = Mag;
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Mag::Unit, Mag::Unit) => Mag::Unit,
            _ => Mag::Any,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MagParam {
    A,
    B,
    C,
    D,
    Mag(Mag),
}

impl From<Mag> for MagParam {
    fn from(value: Mag) -> Self {
        MagParam::Mag(value)
    }
}

impl From<Mag> for TypeParam {
    fn from(value: Mag) -> Self {
        TypeParam::Mag(MagParam::Mag(value))
    }
}

impl ToTokens for MagParam {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            MagParam::Mag(mag) => return mag.to_tokens(tokens),
            MagParam::A => quote!(A),
            MagParam::B => quote!(B),
            MagParam::C => quote!(C),
            MagParam::D => quote!(D),
        }
        .to_tokens(tokens)
    }
}

impl MagParam {
    pub fn is_generic(self) -> bool {
        !matches!(self, Self::Mag(_))
    }

    pub fn mul(self, rhs: Self, output: Self) -> TraitBound {
        TraitBound::Binary(BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Mul,
            rhs: rhs.into(),
            output: Some(output.into()),
        })
    }

    pub fn div(self, rhs: Self, output: Self) -> TraitBound {
        TraitBound::Binary(BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Div,
            rhs: rhs.into(),
            output: Some(output.into()),
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FloatParam {
    T,
    U,
    V,
    Float(Float),
}

impl From<Float> for FloatParam {
    fn from(value: Float) -> Self {
        FloatParam::Float(value)
    }
}

impl ToTokens for FloatParam {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            FloatParam::Float(float) => return float.to_tokens(tokens),
            FloatParam::T => quote!(T),
            FloatParam::U => quote!(U),
            FloatParam::V => quote!(V),
        }
        .to_tokens(tokens)
    }
}

impl FloatParam {
    pub fn is_generic(&self) -> bool {
        !matches!(*self, FloatParam::Float(_))
    }

    pub fn from(self, from: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            rhs: from.into(),
            binary_trait: BinaryTrait::From,
            output: None,
        }
        .into()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Neg,
            output: Some(self.into()),
        }
        .into()
    }

    pub fn sqrt(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Sqrt,
            output: Some(self.into()),
        }
        .into()
    }

    pub fn zero(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Zero,
            output: None,
        }
        .into()
    }

    pub fn zero_const(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::ZeroConst,
            output: None,
        }
        .into()
    }

    pub fn one(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::One,
            output: None,
        }
        .into()
    }

    pub fn one_const(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::OneConst,
            output: None,
        }
        .into()
    }

    pub fn add(self, rhs: FloatParam, output: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Add,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn sub(self, rhs: FloatParam, output: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Sub,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn add_assign(self, rhs: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::AddAssign,
            rhs: rhs.into(),
            output: None,
        }
        .into()
    }

    pub fn sub_assign(self, rhs: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::SubAssign,
            rhs: rhs.into(),
            output: None,
        }
        .into()
    }

    pub fn mul(self, rhs: FloatParam, output: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Mul,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn div(self, rhs: FloatParam, output: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Div,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn partial_eq(self, rhs: FloatParam) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            rhs: rhs.into(),
            binary_trait: BinaryTrait::PartialEq,
            output: None,
        }
        .into()
    }

    pub fn copy(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Copy,
            output: None,
        }
        .into()
    }

    pub fn inv(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Inverse,
            output: None,
        }
        .into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterizedType {
    ty: OverType,
    param: FloatParam,
    mag: MagParam,
}

impl ParameterizedType {
    #[track_caller]
    pub fn new(ty: OverType, param: FloatParam, mag: MagParam) -> Self {
        match (ty, param) {
            (OverType::Float(l), FloatParam::Float(r)) if l == r => {
                ParameterizedType { ty, param, mag }
            }
            (OverType::Float(_), _) => {
                panic!("float type parameter mismatch: {ty:?} and {param:?}")
            }
            (ty, param) => ParameterizedType { ty, param, mag },
        }
    }

    pub fn zero(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Zero,
            output: None,
        }
        .into()
    }

    pub fn one(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::One,
            output: None,
        }
        .into()
    }

    pub fn add(self, rhs: Self, output: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Add,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn sub(self, rhs: Self, output: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Sub,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn mul(self, rhs: Self, output: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Mul,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn div(self, rhs: Self, output: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Div,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn geo(self, rhs: Self, output: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::Geo,
            rhs: rhs.into(),
            output: Some(output.into()),
        }
        .into()
    }

    pub fn inv(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Inverse,
            output: Some(self.into()),
        }
        .into()
    }

    pub fn partial_eq(self, rhs: Self) -> TraitBound {
        BinaryTraitBound {
            lhs: self.into(),
            binary_trait: BinaryTrait::PartialEq,
            rhs: rhs.into(),
            output: None,
        }
        .into()
    }

    pub fn norm2(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Norm2,
            output: Some(
                crate::Type::Grade(0)
                    .with_type_param(self.param, self.mag)
                    .into(),
            ),
        }
        .into()
    }

    pub fn rev(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Reverse,
            output: Some(self.into()),
        }
        .into()
    }

    pub fn copy(self) -> TraitBound {
        UnaryTraitBound {
            ty: self.into(),
            unary_trait: UnaryTrait::Copy,
            output: None,
        }
        .into()
    }
}

impl ToTokens for ParameterizedType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self { ty, param: t, mag } = *self;
        match ty {
            OverType::Type(ty) => quote!(#ty<#t, #mag>),
            OverType::Float(f) => quote!(#f),
        }
        .to_tokens(tokens);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TraitBound {
    Unary(UnaryTraitBound),
    Binary(BinaryTraitBound),
}

impl From<UnaryTraitBound> for TraitBound {
    fn from(value: UnaryTraitBound) -> Self {
        TraitBound::Unary(value)
    }
}

impl From<BinaryTraitBound> for TraitBound {
    fn from(value: BinaryTraitBound) -> Self {
        TraitBound::Binary(value)
    }
}

impl ToTokens for TraitBound {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            TraitBound::Unary(unary) => unary.to_tokens(tokens),
            TraitBound::Binary(binary) => binary.to_tokens(tokens),
        }
    }
}

impl TraitBound {
    pub fn ty(&self) -> TypeParam {
        match self {
            TraitBound::Unary(unary) => unary.ty,
            TraitBound::Binary(binary) => binary.lhs,
        }
    }

    pub fn copy_params(&self) -> impl Iterator<Item = TypeParam> + '_ {
        match self {
            TraitBound::Unary(unary) => unary.copy_params(),
            TraitBound::Binary(binary) => binary.copy_params(),
        }
        .into_iter()
        .flatten()
    }

    pub fn generics(&self) -> impl Iterator<Item = TypeParam> + '_ {
        match self {
            TraitBound::Unary(unary) => [Some(unary.ty), unary.output, None],
            TraitBound::Binary(binary) => [Some(binary.lhs), Some(binary.rhs), binary.output],
        }
        .into_iter()
        .flatten()
        .flat_map(|ty| ty.generics())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct UnaryTraitBound {
    pub ty: TypeParam,
    pub unary_trait: UnaryTrait,
    pub output: Option<TypeParam>,
}

impl ToTokens for UnaryTraitBound {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let UnaryTraitBound {
            unary_trait,
            output,
            ..
        } = self;
        if let Some(output) = output {
            quote!(#unary_trait<Output = #output>)
        } else {
            quote!(#unary_trait)
        }
        .to_tokens(tokens);
    }
}

impl UnaryTraitBound {
    pub fn copy_params(&self) -> [Option<TypeParam>; 2] {
        use UnaryTrait::*;
        match self.unary_trait {
            Norm2 | Norm => [Some(self.ty), None],
            _ => [None; 2],
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinaryTraitBound {
    pub lhs: TypeParam,
    pub binary_trait: BinaryTrait,
    pub rhs: TypeParam,
    pub output: Option<TypeParam>,
}

impl ToTokens for BinaryTraitBound {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let BinaryTraitBound {
            binary_trait,
            rhs,
            output,
            ..
        } = self;
        if let Some(output) = output {
            quote!(#binary_trait<#rhs, Output = #output>).to_tokens(tokens);
        } else {
            quote!(#binary_trait<#rhs>).to_tokens(tokens);
        }
    }
}

impl BinaryTraitBound {
    pub fn copy_params(&self) -> [Option<TypeParam>; 2] {
        use BinaryTrait::*;
        match self.binary_trait {
            Mul | Div => [Some(self.lhs), Some(self.rhs)],
            MulAssign | DivAssign => [Some(self.rhs), None],
            _ => [None; 2],
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Lifetime {
    A,
}

impl ToTokens for Lifetime {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::A => quote!('a).to_tokens(tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_bounds_floats() {
        let mul_different_floats =
            TraitBounds::product_types(OverType::Float(Float::F32), OverType::Float(Float::F64));
        assert!(mul_different_floats.is_none());
    }
}
