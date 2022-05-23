use super::algebra::*;
use itertools::iproduct;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::iter::once;
use syn::token::{Brace, For, Impl};
use syn::{parse_quote, Expr, Generics, ImplItem, ItemImpl};

trait To_ {
    fn to_<U: syn::parse::Parse>(&self) -> U;
}

impl<T: ToTokens> To_ for T {
    fn to_<U: syn::parse::Parse>(&self) -> U {
        let tokens = self.to_token_stream();
        parse_quote!(#tokens)
    }
}

impl Algebra {
    pub fn define_mv(self) -> TokenStream {
        let _traits = if self.is_homogenous() {
            quote! {
                pub use crate::{
                    Dot,
                    Antidot,
                    Wedge,
                    Antiwedge,
                    Commutator,
                    Geometric,
                    Antigeometric,
                    LeftComplement,
                    RightComplement,
                    Reverse,
                    Antireverse,
                    Bulk,
                    Weight,
                    IsIdeal
                };
            }
        } else {
            quote! {
                pub use crate::{
                    Dot,
                    Wedge,
                    Commutator,
                    Geometric,
                    LeftComplement,
                    RightComplement,
                    Reverse
                };
            }
        };

        let grade_products = GradeProducts(self);

        let types = TypeMv::iter(self).map(TypeMv::define);

        let ops = ProductOp::iter_all(self)
            .flat_map(|op| TypeMv::iter(self).map(move |lhs| (op, lhs)))
            .flat_map(|(op, lhs)| TypeMv::iter(self).map(move |rhs| (op, lhs, rhs)))
            .filter_map(|(op, lhs, rhs)| impl_item_for_product_op(op, lhs, rhs));

        quote! {
            // #traits
            #grade_products
            #(#types)*
            #(#ops)*
        }
    }
}

pub fn impl_item_for_product_op(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> Option<ItemImpl> {
    if op.is_std() && lhs.is_scalar() && rhs.is_scalar() {
        return None;
    }

    Some(ItemImpl {
        attrs: vec![],
        defaultness: None,
        unsafety: None,
        impl_token: Impl::default(),
        generics: generics(lhs, rhs, op),
        trait_: {
            let trait_ty = op.ty();
            let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX);
            Some((None, parse_quote!(#trait_ty<#rhs_ty>), For::default()))
        },
        self_ty: Box::new(lhs.ty_with_suffix(LHS_SUFFIX)),
        brace_token: Brace::default(),
        items: {
            let output_type = output_type_item(op, lhs, rhs);
            let trait_fn = trait_fn_item(op, lhs, rhs);
            vec![output_type, trait_fn]
        },
    })
}

pub fn generics(lhs: TypeMv, rhs: TypeMv, op: ProductOp) -> syn::Generics {
    if !is_generic(lhs, rhs) {
        return Default::default();
    }

    let algebra = lhs.algebra();

    let mut generics = Generics::default();
    let mut product_bounds = Vec::<syn::WherePredicate>::default();
    let mut copy_bounds = Vec::<syn::WherePredicate>::default();
    let mut sum_bounds = Vec::<syn::WherePredicate>::default();

    for grade in lhs.generics(LHS_SUFFIX).chain(rhs.generics(RHS_SUFFIX)) {
        generics.params.push(grade.to_());
        copy_bounds.push(parse_quote! { #grade: Copy });
    }

    for grade in algebra.grades() {
        let int = intermediate_types(op, lhs, rhs, grade);
        let sum = intermediate_and_sum_types(op, lhs, rhs, grade);

        for ident in (0..sum).into_iter().map(|n| grade.generic_n(n)) {
            generics.params.push(ident.to_());
        }

        let op_trait = ProductOp::Grade(grade).ty();
        for (n, (lhs_grade, rhs_grade)) in iproduct!(lhs.grades(), rhs.grades())
            .filter(|(lhs, rhs)| op.output_contains(*lhs, *rhs, grade))
            .enumerate()
        {
            let lhs = grade_type(lhs, lhs_grade, LHS_SUFFIX);
            let rhs = grade_type(rhs, rhs_grade, RHS_SUFFIX);
            let out = grade.generic_n(n);
            product_bounds.push(parse_quote!( #lhs: #op_trait<#rhs, Output = #out> ));
        }

        let lhs = once(0).chain(int..sum).map(|n| grade.generic_n(n));
        let rhs = (1..int).into_iter().map(|n| grade.generic_n(n));
        let out = (int..sum).into_iter().map(|n| grade.generic_n(n));
        for ((lhs, rhs), out) in lhs.zip(rhs).zip(out) {
            let predicate = parse_quote! { #lhs: std::ops::Add<#rhs, Output = #out> };
            sum_bounds.push(predicate);
        }
    }

    let where_clause = generics.make_where_clause();
    where_clause.predicates.extend(product_bounds);
    where_clause.predicates.extend(copy_bounds);
    where_clause.predicates.extend(sum_bounds);

    generics
}

fn is_generic(lhs: TypeMv, rhs: TypeMv) -> bool {
    lhs.is_generic() || rhs.is_generic()
}

fn last_generics(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> impl Iterator<Item = Ident> {
    lhs.algebra()
        .grades()
        .map(move |grade| last_generic(op, lhs, rhs, grade))
}

fn last_generic(op: ProductOp, lhs: TypeMv, rhs: TypeMv, grade: Grade) -> Ident {
    let int = intermediate_and_sum_types(op, lhs, rhs, grade);
    let last = int.checked_sub(1);
    last.map(|n| grade.generic_n(n))
        .unwrap_or_else(|| Zero::ident())
}

fn intermediate_and_sum_types(op: ProductOp, lhs: TypeMv, rhs: TypeMv, grade: Grade) -> usize {
    let int = intermediate_types(op, lhs, rhs, grade);
    (int * 2).checked_sub(1).unwrap_or_default()
}

fn intermediate_types(op: ProductOp, lhs: TypeMv, rhs: TypeMv, grade: Grade) -> usize {
    iproduct!(lhs.grades(), rhs.grades())
        .filter(|(lhs, rhs)| op.output_contains(*lhs, *rhs, grade))
        .count()
}

fn output_type_item(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> ImplItem {
    let generic = is_generic(lhs, rhs);
    let output_ty = match op.output_mv(lhs, rhs) {
        TypeMv::Zero(_) => Zero::ty(),
        TypeMv::Grade(grade) => {
            if generic {
                last_generic(op, lhs, rhs, grade).to_()
            } else {
                grade.ty()
            }
        }
        TypeMv::Multivector(mv) => {
            let ty = Multivector::ident();
            if generic {
                let types = last_generics(op, lhs, rhs);
                parse_quote!( #ty <#(#types),*> )
            } else {
                let types = mv.type_parameters("Out");
                parse_quote!( #ty <#(#types),*> )
            }
        }
    };
    parse_quote! { type Output = #output_ty; }
}

fn trait_fn_item(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> ImplItem {
    let op_fn = op.fn_ident();
    let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX);
    let output_expr = output_expr(op, lhs, rhs);
    parse_quote! {
        #[inline]
        #[allow(unused_variables)]
        fn #op_fn(self, rhs: #rhs_ty) -> Self::Output {
            #output_expr
        }
    }
}

fn output_expr(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> Expr {
    let output = op.output_mv(lhs, rhs);
    match output {
        TypeMv::Zero(_) => Zero::expr(),
        TypeMv::Grade(grade) => {
            if is_generic(lhs, rhs) {
                grade_sum_expr(op, lhs, rhs, grade)
            } else {
                grade_fields_expr(op, grade, lhs, rhs)
            }
        }
        TypeMv::Multivector(mv) => mv_expr(mv, lhs, rhs, op),
    }
}

fn grade_sum_expr(op: ProductOp, lhs: TypeMv, rhs: TypeMv, grade: Grade) -> Expr {
    let products =
        iproduct!(lhs.grades(), rhs.grades()).filter_map::<Expr, _>(|(lhs_grade, rhs_grade)| {
            grade_product_expr(grade, lhs, lhs_grade, rhs, rhs_grade, op)
        });
    parse_quote! { #(#products)+* }
}

fn grade_fields_expr(op: ProductOp, grade: Grade, lhs: TypeMv, rhs: TypeMv) -> syn::Expr {
    let fields = grade.blades().map(|blade| {
        let sum = cartesian_product(lhs, rhs, op)
            .filter_map(|(lhs_blade, rhs_blade, _)| op.expr(lhs, lhs_blade, rhs, rhs_blade, blade))
            .collect();
        assign_field(TypeMv::Grade(grade), blade, &sum)
    });

    if grade.is_scalar() {
        parse_quote! { #(#fields)+* }
    } else {
        let ty = grade.ty();
        parse_quote! {
            #ty {
                #( #fields, )*
            }
        }
    }
}

const LHS_SUFFIX: &'static str = "Lhs";
const RHS_SUFFIX: &'static str = "Rhs";

pub fn mv_expr(mv: Multivector, lhs: TypeMv, rhs: TypeMv, op: ProductOp) -> Expr {
    let algebra = mv.1;
    let grades = algebra
        .grades()
        .map(|grade| {
            let sum = iproduct!(lhs.grades(), rhs.grades())
                .filter_map(|(lhs_grade, rhs_grade)| {
                    grade_product_expr(grade, lhs, lhs_grade, rhs, rhs_grade, op)
                })
                .collect();
            (grade, sum)
        })
        .collect();
    OutputExpr::Multivector(grades).expr()
}

fn grade_type(ty: TypeMv, grade: Grade, suffix: &str) -> syn::Type {
    if ty.is_generic() {
        grade.generic(suffix).to_()
    } else {
        grade.ty()
    }
}

pub fn grade_product_expr(
    grade: Grade,
    lhs: TypeMv,
    lhs_grade: Grade,
    rhs: TypeMv,
    rhs_grade: Grade,
    op: ProductOp,
) -> Option<Expr> {
    if op.output_contains(lhs_grade, rhs_grade, grade) {
        let op_fn = ProductOp::Grade(grade).fn_ident();
        let lhs = access_grade(lhs, lhs_grade, quote!(self));
        let rhs = access_grade(rhs, rhs_grade, quote!(rhs));
        Some(parse_quote! { #lhs.#op_fn(#rhs) })
    } else {
        None
    }
}

pub enum OutputExpr {
    Multivector(Vec<(Grade, Vec<syn::Expr>)>),
}

impl OutputExpr {
    pub fn expr(&self) -> syn::Expr {
        match self {
            Self::Multivector(grades) => {
                let ident = Multivector::ident();
                let algebra = grades.iter().next().unwrap().0 .1;
                let fields = algebra.grades().map(|grade| {
                    if let Some((_, sum)) = grades.iter().find(|(g, _)| *g == grade) {
                        if sum.is_empty() {
                            Zero::expr()
                        } else {
                            parse_quote! {
                                #( #sum )+*
                            }
                        }
                    } else {
                        Zero::expr()
                    }
                });
                parse_quote! {
                    #ident(
                        #(#fields),*
                    )
                }
            }
        }
    }
}

impl ToTokens for OutputExpr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.expr().to_tokens(tokens);
    }
}

fn access_grade(parent: TypeMv, grade: Grade, ident: TokenStream) -> syn::Expr {
    let field = match parent {
        TypeMv::Grade(_) => return parse_quote! { #ident },
        TypeMv::Multivector(_) => grade.mv_field(),
        TypeMv::Zero(_) => unreachable!("no grades to access"),
    };
    parse_quote! {
        #ident.#field
    }
}

fn assign_field(output: TypeMv, blade: Blade, sum: &Vec<syn::Expr>) -> TokenStream {
    let expr = if sum.is_empty() {
        quote! { 0. }
    } else {
        quote! { #( #sum )+* }
    };
    if output.is_scalar() {
        expr
    } else {
        let field = blade.field();
        quote! { #field: #expr }
    }
}

pub struct GradeProducts(Algebra);

impl ToTokens for GradeProducts {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for grade in self.0.grades() {
            let trait_ident = Ident::new(&format!("{}Product", grade.name()), Span::mixed_site());
            let trait_fn = Ident::new(
                &format!("{}_prod", grade.name().to_lowercase()),
                Span::mixed_site(),
            );

            tokens.extend(quote! {
                pub trait #trait_ident<Rhs> {
                    type Output;
                    fn #trait_fn(self, rhs: Rhs) -> Self::Output;
                }
            });
        }
    }
}

pub struct GradeTypes(Algebra);

impl ToTokens for GradeTypes {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for grade in self.0.grades() {
            if grade.is_scalar() {
                continue;
            }

            let ty = grade.ty();

            let fields = grade.blades().map(|blade| {
                let f = blade.field();
                quote! { pub #f: f64, }
            });

            tokens.extend(quote! {
                #[derive(Debug, Default, Copy, Clone, PartialEq)]
                pub struct #ty {
                    #(#fields)*
                }
            })
        }
    }
}

impl TypeMv {
    fn define(self) -> TokenStream {
        match self {
            Self::Zero(_) => quote! {
                #[derive(Debug, Default, Copy, Clone, PartialEq)]
                pub struct Zero;
            },
            Self::Grade(g) if g.is_scalar() => {
                quote! {}
            }
            Self::Grade(grade) => {
                let ty = grade.ty();
                let fields = grade.blades().map(|blade| {
                    let f = blade.field();
                    quote! { pub #f: f64, }
                });
                quote! {
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct #ty {
                        #(#fields)*
                    }
                }
            }
            Self::Multivector(mv) => {
                let generics = mv.type_parameters("");

                let fields = generics.iter().map(|g| {
                    quote! {
                        pub #g,
                    }
                });

                quote! {
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct Multivector < #(#generics),* > ( #(#fields)* );
                }
            }
        }
    }

    pub fn ty_with_suffix(&self, suffix: &str) -> syn::Type {
        match self {
            Self::Zero(_) => Zero::ty(),
            Self::Grade(g) => g.ty(),
            Self::Multivector(mv) => {
                let ty = Multivector::ident();
                let generics = mv.type_parameters(suffix);
                parse_quote! {
                    #ty < #(#generics),* >
                }
            }
        }
    }
}

impl Multivector {
    const NAME: &'static str = "Multivector";

    pub fn ident() -> syn::Ident {
        syn::parse_str(Self::NAME).unwrap()
    }
}

impl Grade {
    pub fn generic(&self, suffix: &str) -> Ident {
        let g = self.0;
        Ident::new(&format!("G{g}{suffix}"), Span::mixed_site())
    }

    pub fn generic_n(&self, n: usize) -> Ident {
        let g = self.0;
        Ident::new(&format!("G{g}_{n}"), Span::mixed_site())
    }

    pub fn mv_field(&self) -> syn::Member {
        syn::Member::Unnamed(syn::Index::from(self.0 as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn write_to_file(tokens: &impl ToTokens) {
        let contents = tokens.to_token_stream().to_string();
        let file = "../output.rs";
        std::fs::write(file, contents).unwrap();
    }

    // #[test]
    // fn impl_g3() {
    //     let g3 = Algebra::new(3, 0, 0).define_mv();
    //     write_to_file(&g3);
    // }

    fn assert_eq_impl(expected: &ItemImpl, actual: &ItemImpl) {
        assert_eq_slice(&expected.attrs, &actual.attrs, "attrs");
        assert_eq_ref(&expected.defaultness, &actual.defaultness, "default");
        assert_eq_ref(&expected.unsafety, &actual.unsafety, "unsafety");
        assert_eq_ref(&expected.impl_token, &actual.impl_token, "impl_token");

        match (&expected.trait_, &actual.trait_) {
            (Some(te), Some(ta)) => {
                assert_eq_ref(&te.1, &ta.1, "trait");
            }
            (None, None) => {}
            _ => panic!("ItemImpl.trait_ are not eq"),
        }

        assert_eq_slice(&expected.items, &actual.items, "items");

        assert_eq_generics(&expected.generics, &actual.generics);
    }

    fn assert_eq_ref<T: ToTokens>(expected: &T, actual: &T, message: &str) {
        assert_eq!(
            expected.to_token_stream().to_string(),
            actual.to_token_stream().to_string(),
            "{}",
            message
        );
    }

    fn assert_eq_slice<T: ToTokens>(expected: &[T], actual: &[T], message: &str) {
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(
                expected.to_token_stream().to_string(),
                actual.to_token_stream().to_string(),
                "{}",
                message
            );
        }
    }

    fn assert_eq_generics(expected: &Generics, actual: &Generics) {
        let (ea, eb, ec) = expected.split_for_impl();
        let (aa, ab, ac) = actual.split_for_impl();
        assert_eq_ref(&ea, &aa, "impl generics");
        assert_eq_ref(&eb, &ab, "type generics");
        assert_eq_ref(&ec, &ac, "where clause");
    }

    #[test]
    fn mv_mv_bivector_product() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = TypeMv::Multivector(Multivector::new(algebra));

        let impl_item =
            impl_item_for_product_op(ProductOp::Grade(algebra.grade(2)), mv, mv).unwrap();
        let expected: ItemImpl = parse_quote! {
            impl < G0Lhs , G1Lhs , G2Lhs , G3Lhs , G0Rhs , G1Rhs , G2Rhs , G3Rhs , G2_0 , G2_1 , G2_2 , G2_3 , G2_4 , G2_5 , G2_6 , G2_7 , G2_8 , G2_9 , G2_10 > BivectorProduct < Multivector < G0Rhs , G1Rhs , G2Rhs , G3Rhs > > for Multivector < G0Lhs , G1Lhs , G2Lhs , G3Lhs > where G0Lhs : BivectorProduct < G2Rhs , Output = G2_0 > , G1Lhs : BivectorProduct < G1Rhs , Output = G2_1 > , G1Lhs : BivectorProduct < G3Rhs , Output = G2_2 > , G2Lhs : BivectorProduct < G0Rhs , Output = G2_3 > , G2Lhs : BivectorProduct < G2Rhs , Output = G2_4 > , G3Lhs : BivectorProduct < G1Rhs , Output = G2_5 > , G0Lhs : Copy , G1Lhs : Copy , G2Lhs : Copy , G3Lhs : Copy , G0Rhs : Copy , G1Rhs : Copy , G2Rhs : Copy , G3Rhs : Copy , G2_0 : std :: ops :: Add < G2_1 , Output = G2_6 > , G2_6 : std :: ops :: Add < G2_2 , Output = G2_7 > , G2_7 : std :: ops :: Add < G2_3 , Output = G2_8 > , G2_8 : std :: ops :: Add < G2_4 , Output = G2_9 > , G2_9 : std :: ops :: Add < G2_5 , Output = G2_10 > { type Output = G2_10 ; # [inline] # [allow (unused_variables)] fn bivector_prod (self , rhs : Multivector < G0Rhs , G1Rhs , G2Rhs , G3Rhs >) -> Self :: Output { self . 0 . bivector_prod (rhs . 2) + self . 1 . bivector_prod (rhs . 1) + self . 1 . bivector_prod (rhs . 3) + self . 2 . bivector_prod (rhs . 0) + self . 2 . bivector_prod (rhs . 2) + self . 3 . bivector_prod (rhs . 1) } }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    fn mv_mv_product() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = TypeMv::Multivector(Multivector::new(algebra));

        let impl_item = impl_item_for_product_op(ProductOp::Mul, mv, mv).unwrap();
        let expected = parse_quote! {
            impl < G0Lhs , G1Lhs , G2Lhs , G0Rhs , G1Rhs , G2Rhs , G0_0 , G0_1 , G0_2 , G0_3 , G0_4 , G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G2_0 , G2_1 , G2_2 , G2_3 , G2_4 > std :: ops :: Mul < Multivector < G0Rhs , G1Rhs , G2Rhs > > for Multivector < G0Lhs , G1Lhs , G2Lhs > where G0Lhs : ScalarProduct < G0Rhs , Output = G0_0 > , G1Lhs : ScalarProduct < G1Rhs , Output = G0_1 > , G2Lhs : ScalarProduct < G2Rhs , Output = G0_2 > , G0Lhs : VectorProduct < G1Rhs , Output = G1_0 > , G1Lhs : VectorProduct < G0Rhs , Output = G1_1 > , G1Lhs : VectorProduct < G2Rhs , Output = G1_2 > , G2Lhs : VectorProduct < G1Rhs , Output = G1_3 > , G0Lhs : BivectorProduct < G2Rhs , Output = G2_0 > , G1Lhs : BivectorProduct < G1Rhs , Output = G2_1 > , G2Lhs : BivectorProduct < G0Rhs , Output = G2_2 > , G0Lhs : Copy , G1Lhs : Copy , G2Lhs : Copy , G0Rhs : Copy , G1Rhs : Copy , G2Rhs : Copy , G0_0 : std :: ops :: Add < G0_1 , Output = G0_3 > , G0_3 : std :: ops :: Add < G0_2 , Output = G0_4 > , G1_0 : std :: ops :: Add < G1_1 , Output = G1_4 > , G1_4 : std :: ops :: Add < G1_2 , Output = G1_5 > , G1_5 : std :: ops :: Add < G1_3 , Output = G1_6 > , G2_0 : std :: ops :: Add < G2_1 , Output = G2_3 > , G2_3 : std :: ops :: Add < G2_2 , Output = G2_4 > { type Output = Multivector < G0_4 , G1_6 , G2_4 > ; # [inline] # [allow (unused_variables)] fn mul (self , rhs : Multivector < G0Rhs , G1Rhs , G2Rhs >) -> Self :: Output { Multivector (self . 0 . scalar_prod (rhs . 0) + self . 1 . scalar_prod (rhs . 1) + self . 2 . scalar_prod (rhs . 2) , self . 0 . vector_prod (rhs . 1) + self . 1 . vector_prod (rhs . 0) + self . 1 . vector_prod (rhs . 2) + self . 2 . vector_prod (rhs . 1) , self . 0 . bivector_prod (rhs . 2) + self . 1 . bivector_prod (rhs . 1) + self . 2 . bivector_prod (rhs . 0)) } }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    fn vector_product_of_two_vectors_is_zero() {
        let algebra = Algebra::new(2, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));

        // <vector * vector>_1 = 0
        let impl_item =
            impl_item_for_product_op(ProductOp::Grade(algebra.grade(1)), vector, vector).unwrap();
        let expected = parse_quote! {
            impl VectorProduct < Vector > for Vector { type Output = Zero ; # [inline] # [allow (unused_variables)] fn vector_prod (self , rhs : Vector) -> Self :: Output { Zero } }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    fn vector_product_of_vectors_and_bivector_is_vector() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));
        let bivector = TypeMv::Grade(algebra.grade(2));

        // <vector * bivector>_1 = vector
        let impl_item =
            impl_item_for_product_op(ProductOp::Grade(algebra.grade(1)), vector, bivector).unwrap();
        let expected = parse_quote! {
            impl VectorProduct < Bivector > for Vector { type Output = Vector ; # [inline] # [allow (unused_variables)] fn vector_prod (self , rhs : Bivector) -> Self :: Output { Vector { e1 : - (self . e2 * rhs . e12) + - (self . e3 * rhs . e13) , e2 : self . e1 * rhs . e12 + - (self . e3 * rhs . e23) , e3 : self . e1 * rhs . e13 + self . e2 * rhs . e23 , } } }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    fn vector_vector_product() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));

        let impl_item = impl_item_for_product_op(ProductOp::Mul, vector, vector).unwrap();
        let expected = parse_quote! {
            impl std::ops::Mul<Vector> for Vector {
                type Output = Multivector<f64, Zero, Bivector, Zero>;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Vector) -> Self::Output {
                    Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
                }
            }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    fn scalar_scalar_geo() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = TypeMv::Grade(algebra.grade(0));

        let impl_item = impl_item_for_product_op(ProductOp::Geometric, scalar, scalar).unwrap();
        let expected = parse_quote! {
            impl crate::Geometric<f64> for f64 {
                type Output = f64;
                #[inline]
                #[allow(unused_variables)]
                fn geo(self, rhs: f64) -> Self::Output {
                    self * rhs
                }
            }
        };

        assert_eq_impl(&expected, &impl_item);
    }

    #[test]
    pub fn scalar_mv_scalar_product() {
        let g2 = Algebra::new(2, 0, 0);
        let scalar = g2.grade(0);

        let actual = impl_item_for_product_op(
            ProductOp::Grade(scalar),
            TypeMv::Grade(scalar),
            TypeMv::Multivector(g2.mv()),
        )
        .unwrap();

        let expected = parse_quote! {
            impl<G0Rhs, G1Rhs, G2Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs>> for f64
            where
                f64: ScalarProduct<G0Rhs, Output = G0_0>,
                G0Rhs: Copy,
                G1Rhs: Copy,
                G2Rhs: Copy
            {
                type Output = G0_0;
                #[inline]
                #[allow(unused_variables)]
                fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs>) -> Self::Output {
                    self.scalar_prod(rhs.0)
                }
            }
        };

        assert_eq_impl(&expected, &actual);
    }

    #[test]
    pub fn scalar_mv_mul() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = algebra.grade(0);

        let actual = impl_item_for_product_op(
            ProductOp::Mul,
            TypeMv::Grade(scalar),
            TypeMv::Multivector(algebra.mv()),
        )
        .unwrap();

        let expected = parse_quote! {
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
                G3Rhs: Copy
            {
                type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
                    Multivector(
                        self.scalar_prod(rhs.0),
                        self.vector_prod(rhs.1),
                        self.bivector_prod(rhs.2),
                        self.trivector_prod(rhs.3)
                    )
                }
            }
        };

        assert_eq_impl(&expected, &actual);
    }

    #[test]
    fn mul_scalars_is_none() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = TypeMv::Grade(algebra.grade(0));
        assert!(impl_item_for_product_op(ProductOp::Mul, scalar, scalar).is_none());
    }

    #[test]
    fn intermediate_type_test() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = TypeMv::Multivector(algebra.mv());
        let bivector = algebra.grade(2);
        let trivector = algebra.grade(3);

        assert_eq!(4, intermediate_types(ProductOp::Mul, mv, mv, trivector));

        assert_eq!(
            1,
            intermediate_types(ProductOp::Mul, mv, TypeMv::Grade(bivector), trivector)
        );

        assert_eq!(6, intermediate_types(ProductOp::Mul, mv, mv, bivector));
    }

    #[test]
    fn last_type_test() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = TypeMv::Multivector(algebra.mv());
        let bivector = algebra.grade(2);

        assert_eq_ref(
            &bivector.generic_n(10),
            &last_generic(ProductOp::Mul, mv, mv, bivector),
            "ident",
        );

        assert_eq_ref(
            &Zero::ident(),
            &last_generic(ProductOp::Grade(algebra.grade(0)), mv, mv, bivector),
            "ident",
        );
    }

    #[test]
    fn mv_generics_test() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = TypeMv::Multivector(algebra.mv());
        let op = ProductOp::Mul;
        let item_impl = impl_item_for_product_op(op, mv, mv).unwrap();

        let expected = item_impl.generics;
        let actual = generics(mv, mv, op);

        assert_eq!(
            expected.where_clause.as_ref().unwrap().predicates.len(),
            actual.where_clause.as_ref().unwrap().predicates.len()
        );

        assert_eq_generics(&expected, &actual);
    }
}
