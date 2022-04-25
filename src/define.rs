use super::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let traits = traits();
        let zero = zero();

        let blades = self
            .blades()
            .filter(|b| !b.0.is_empty())
            .map(|b| b.define());

        let grades = self.grades_without_scalar().map(|g| g.define());

        let even = SubAlgebra::Even(*self).define();
        let odd = SubAlgebra::Odd(*self).define();

        quote! {
            #traits

            #zero

            #(
                #blades
            )*

            #(
                #grades
            )*

            #even

            #odd
        }
    }
}

fn zero() -> TokenStream {
    quote! {
        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
        pub struct Zero;

        impl<T> std::ops::Add<T> for Zero {
            type Output = T;
            fn add(self, rhs: T) -> T {
                rhs
            }
        }

        impl<T: std::ops::Neg<Output = T>> std::ops::Sub<T> for Zero {
            type Output = T;
            fn sub(self, rhs: T) -> T {
                -rhs
            }
        }

        impl<T> std::ops::Mul<T> for Zero {
            type Output = Zero;
            fn mul(self, _rhs: T) -> Zero {
                Zero
            }
        }
    }
}

fn traits() -> TokenStream {
    quote! {
        pub trait Dot<Rhs> {
            type Output;
            fn dot(self, rhs: Rhs) -> Self::Output;
        }

        pub trait Commutator<Rhs> {
            type Output;
            fn commutator(self, rhs: Rhs) -> Self::Output;
        }

        pub trait Wedge<Rhs> {
            type Output;
            fn wedge(self, rhs: Rhs) -> Self::Output;
        }
    }
}

impl ToTokens for Multiplier {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let t = match self {
            Multiplier::One => quote! { f64 },
            Multiplier::Zero => quote! { Zero },
            Multiplier::NegOne => quote! { f64 },
        };
        tokens.extend(t);
    }
}

impl Blade {
    pub fn define(&self) -> proc_macro2::TokenStream {
        let ty = self.type_ident();

        let mul_blades = self.1.blades().filter(|b| b.0 .0 != 0).map(|rhs| {
            let rhs_ty = rhs.type_ident();
            let (multiplier, output) = *self * rhs;

            let output_ty = match multiplier {
                Multiplier::Zero => quote! { Zero },
                _ => {
                    let output_ty = output.type_ident();
                    quote! { #output_ty }
                }
            };

            let expr = match (multiplier, output) {
                (Multiplier::Zero, _) => quote! { Zero },
                (Multiplier::One, b) => {
                    if b == self.1.scalar() {
                        quote! { self.0 * rhs.0 }
                    } else {
                        quote! { #output_ty(self.0 * rhs.0) }
                    }
                }
                (Multiplier::NegOne, b) => {
                    if b == self.1.scalar() {
                        quote! { -(self.0 * rhs.0) }
                    } else {
                        quote! { #output_ty(-(self.0 * rhs.0)) }
                    }
                }
            };

            let rhs_ident = match multiplier {
                Multiplier::Zero => quote! { _ },
                _ => quote! { rhs },
            };

            quote! {
                impl std::ops::Mul<#rhs_ty> for #ty {
                    type Output = #output_ty;
                    fn mul(self, #rhs_ident: #rhs_ty) -> Self::Output {
                        #expr
                    }
                }
            }
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
            pub struct #ty(f64);

            impl std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> #ty {
                    Self(-self.0)
                }
            }

            impl std::ops::Mul<f64> for #ty {
                type Output = #ty;
                fn mul(self, rhs: f64) -> #ty {
                    #ty(self.0 * rhs)
                }
            }

            impl std::ops::Mul<#ty> for f64 {
                type Output = #ty;
                fn mul(self, rhs: #ty) -> #ty {
                    #ty(self * rhs.0)
                }
            }

            impl std::ops::Mul<Zero> for #ty {
                type Output = Zero;
                fn mul(self, rhs: Zero) -> Zero {
                    rhs
                }
            }

            #(#mul_blades)*

            impl std::ops::Add<Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: Zero) -> #ty {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: Zero) -> #ty {
                    self
                }
            }

            impl std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: #ty) -> #ty {
                    #ty(self.0 + rhs.0)
                }
            }

            impl std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: #ty) -> #ty {
                    #ty(self.0 - rhs.0)
                }
            }
        }
    }

    pub fn field(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("scalar", Span::mixed_site())
        } else {
            let mut output = "e".to_string();

            for i in 1..=self.1.sum() {
                if self.0.contains(i) {
                    output.push_str(&i.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }

    pub fn type_ident(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("f64", Span::mixed_site())
        } else {
            let mut output = "E".to_string();

            for basis in self.1.bases() {
                if self.0.contains(basis.0) {
                    output.push_str(&basis.0.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }
}

impl Grade {
    pub fn define(&self) -> TokenStream {
        let ty = self.type_ident();

        let struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let new_fn_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: f64 }
        });

        let new_fn_struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { #f: #ty(#f), }
        });

        let neg_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: -self.#f, }
        });

        let add_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f + rhs.#f, }
        });

        let type_ops = self.algebra().types().into_iter().flat_map(|t| {
            ProductOp::iter().map(move |op| {
                MulProduct {
                    lhs: *self,
                    rhs: t,
                    op,
                }
                .implement()
            })
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #(#struct_fields)*
            }

            impl #ty {
                pub fn new(
                    #(#new_fn_fields,)*
                ) -> Self {
                    Self {
                        #(#new_fn_struct_fields)*
                    }
                }
            }

            impl std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> Self {
                    Self {
                        #(#neg_fields)*
                    }
                }
            }

            impl std::ops::Add for #ty {
                type Output = Self;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #(#add_self_fields)*
                    }
                }
            }

            impl std::ops::Add<Zero> for #ty {
                type Output = Self;
                fn add(self, _rhs: Zero) -> Self {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = Self;
                fn sub(self, _rhs: Zero) -> Self {
                    self
                }
            }

            #(#type_ops)*
        }
    }

    pub fn type_ident(&self) -> Ident {
        let str = match self.0 {
            0 => "f64",
            1 => "Vector",
            2 => "Bivector",
            3 => "Trivector",
            4 => "Quadvector",
            5 => "Pentavector",
            6 => "Hexavector",
            7 => "Heptavector",
            8 => "Octovector",
            9 => "Nonavector",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl SubAlgebra {
    pub fn define(&self) -> TokenStream {
        let ty = match self {
            SubAlgebra::Even(_) => quote! { Even },
            SubAlgebra::Odd(_) => quote! { Odd },
        };

        let blades = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let neg_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: -self.#f, }
        });

        let add_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f + rhs.#f, }
        });

        let sub_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f - rhs.#f, }
        });

        let type_ops = self.algebra().types().into_iter().flat_map(|t| {
            ProductOp::iter().map(move |op| {
                MulProduct {
                    lhs: *self,
                    rhs: t,
                    op,
                }
                .implement()
            })
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #( #blades )*
            }

            impl std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> #ty {
                    Self {
                        #(#neg_fields)*
                    }
                }
            }

            impl std::ops::Add<Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: Zero) -> Self::Output {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: Zero) -> Self::Output {
                    self
                }
            }

            impl std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #( #add_self_fields )*
                    }
                }
            }

            impl std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: Self) -> Self {
                    Self {
                        #( #sub_self_fields )*
                    }
                }
            }

            #( #type_ops )*
        }
    }

    fn type_ident(&self) -> Ident {
        let str = match self {
            SubAlgebra::Even(_) => "Even",
            SubAlgebra::Odd(_) => "Odd",
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl Type {
    pub fn iter(alg: Algebra) -> Vec<Self> {
        let mut vec = vec![Type::Zero(alg)];

        vec.extend(alg.grades_without_scalar().map(Type::Grade));
        vec.extend(alg.subalgebras().map(Type::SubAlgebra));

        vec
    }

    pub fn type_ident(&self) -> Ident {
        match self {
            Type::Zero(_) => Ident::new("Zero", Span::mixed_site()),
            Type::Grade(grade) => grade.type_ident(),
            Type::SubAlgebra(sub) => sub.type_ident(),
        }
    }
}

pub trait CompoundType {
    fn blades(&self) -> Vec<Blade>;
    fn type_ident(&self) -> proc_macro2::Ident;
    fn algebra(&self) -> Algebra;
}

impl CompoundType for Grade {
    fn blades(&self) -> Vec<Blade> {
        self.blades().collect()
    }

    fn type_ident(&self) -> Ident {
        self.type_ident()
    }

    fn algebra(&self) -> Algebra {
        self.1
    }
}

impl CompoundType for SubAlgebra {
    fn blades(&self) -> Vec<Blade> {
        self.blades().collect()
    }

    fn type_ident(&self) -> Ident {
        self.type_ident()
    }

    fn algebra(&self) -> Algebra {
        match self {
            SubAlgebra::Even(alg) => *alg,
            SubAlgebra::Odd(alg) => *alg,
        }
    }
}

impl CompoundType for Type {
    fn blades(&self) -> Vec<Blade> {
        match self {
            Type::Zero(_) => vec![],
            Type::Grade(grade) => grade.blades().collect(),
            Type::SubAlgebra(sub) => sub.blades().collect(),
        }
    }

    fn type_ident(&self) -> Ident {
        self.type_ident()
    }

    fn algebra(&self) -> Algebra {
        match self {
            Type::Zero(alg) => *alg,
            Type::Grade(grade) => grade.1,
            Type::SubAlgebra(sub) => *sub.algebra(),
        }
    }
}

pub struct MulProduct<A, B> {
    pub lhs: A,
    pub rhs: B,
    pub op: ProductOp,
}

pub enum ProductOp {
    Mul,
    Dot,
    Wedge,
}

impl ProductOp {
    pub fn iter() -> impl Iterator<Item = Self> + 'static {
        [Self::Mul, Self::Dot, Self::Wedge].into_iter()
    }

    pub fn call(&self, lhs: Blade, rhs: Blade) -> (Multiplier, Blade) {
        match self {
            ProductOp::Mul => lhs * rhs,
            ProductOp::Dot => lhs.dot(rhs),
            ProductOp::Wedge => lhs.wedge(rhs),
        }
    }

    pub fn trait_ty(&self) -> TokenStream {
        match self {
            ProductOp::Mul => quote! { std::ops::Mul },
            ProductOp::Dot => quote! { Dot },
            ProductOp::Wedge => quote! { Wedge },
        }
    }

    pub fn trait_fn(&self) -> TokenStream {
        match self {
            ProductOp::Mul => quote! { mul },
            ProductOp::Dot => quote! { dot },
            ProductOp::Wedge => quote! { wedge },
        }
    }
}

impl<A, B> MulProduct<A, B>
where
    A: CompoundType,
    B: CompoundType,
{
    fn output(&self) -> Type {
        let iter = self
            .lhs
            .blades()
            .into_iter()
            .flat_map(|lhs_b| {
                self.rhs
                    .blades()
                    .into_iter()
                    .map(move |rhs_b| self.op.call(lhs_b, rhs_b))
            })
            .filter_map(|(multiplier, blade)| {
                if matches!(multiplier, Multiplier::Zero) {
                    None
                } else {
                    Some(blade)
                }
            });
        Type::from_iter(iter, self.lhs.algebra())
    }

    fn expr(&self) -> TokenStream {
        let (ty, blades) = match self.output() {
            Type::Zero(_) => return quote! { Zero },
            Type::Grade(grade) => {
                let blades = grade.blades().collect::<Vec<_>>();
                if grade.0 == 0 {
                    (None, blades)
                } else {
                    let ty = grade.type_ident();
                    (Some(ty), blades)
                }
            }
            Type::SubAlgebra(sub) => {
                let ty = sub.type_ident();
                let blades = sub.blades().collect::<Vec<_>>();
                (Some(ty), blades)
            }
        };

        let ty_some = ty.is_some();
        let fields = blades.into_iter().map(|b| {
            let f = b.field();

            let products = self
                .lhs
                .blades()
                .into_iter()
                .flat_map(|lhs_b| {
                    self.rhs.blades().into_iter().map(move |rhs_b| {
                        let (product, blade) = self.op.call(lhs_b, rhs_b);
                        (lhs_b, rhs_b, product, blade)
                    })
                })
                .filter(|(_, _, multiplier, blade)| *multiplier != Multiplier::Zero && *blade == b)
                .map(|(lhs_b, rhs_b, _, _)| {
                    let lhs_f = lhs_b.field();
                    let rhs_f = rhs_b.field();
                    quote! { self.#lhs_f * rhs.#rhs_f }
                })
                .collect::<syn::punctuated::Punctuated<_, syn::token::Add>>();

            if ty_some {
                if products.is_empty() {
                    if ty_some {
                        quote! { #f: Default::default(), }
                    } else {
                        quote! { Default::default(), }
                    }
                } else {
                    if ty_some {
                        quote! { #f: #products, }
                    } else {
                        quote! { #products }
                    }
                }
            } else {
                quote! { #products }
            }
        });

        if let Some(ty) = ty {
            quote! {
                #ty {
                    #( #fields )*
                }
            }
        } else {
            quote! {
                #( #fields )*
            }
        }
    }

    pub fn implement(&self) -> TokenStream {
        let ty = self.lhs.type_ident();
        let rhs_ty = self.rhs.type_ident();
        let trait_ty = self.op.trait_ty();
        let trait_fn = self.op.trait_fn();
        let output_ty = self.output().type_ident();
        let expr = self.expr();
        let rhs_ident = match self.output() {
            Type::Zero(_) => quote! { _ },
            _ => quote! { rhs },
        };
        quote! {
            impl #trait_ty<#rhs_ty> for #ty {
                type Output = #output_ty;
                fn #trait_fn(self, #rhs_ident: #rhs_ty) -> Self::Output {
                    #expr
                }
            }
        }
    }
}
