"""Tests for core classes: Equation, EquationSystem, and Process composition"""

import pytest
import sympy as sp
from jaco.equation import Equation
from jaco.equation_system import EquationSystem
from jaco.process import Process
from jaco.processes.thermal_process import ThermalProcess
from jaco.symbols import T, n_, d_dt, x_


# ============================================================
# Equation tests
# ============================================================


class TestEquation:
    def test_add_same_lhs(self):
        """Adding two equations with the same LHS sums the RHS"""
        lhs = d_dt(n_("H"))
        eq1 = Equation(lhs, sp.Symbol("a"))
        eq2 = Equation(lhs, sp.Symbol("b"))
        result = eq1 + eq2
        assert result.lhs == lhs
        assert result.rhs == sp.Symbol("a") + sp.Symbol("b")

    def test_sub_same_lhs(self):
        """Subtracting two equations with the same LHS subtracts the RHS"""
        lhs = d_dt(n_("H"))
        eq1 = Equation(lhs, sp.Symbol("a"))
        eq2 = Equation(lhs, sp.Symbol("b"))
        result = eq1 - eq2
        assert result.rhs == sp.Symbol("a") - sp.Symbol("b")

    def test_add_expression(self):
        """Adding a plain sympy expression to an equation adds it to the RHS"""
        lhs = d_dt(n_("H"))
        eq = Equation(lhs, sp.Symbol("a"))
        result = eq + sp.Symbol("b")
        assert result.rhs == sp.Symbol("a") + sp.Symbol("b")
        assert result.lhs == lhs

    def test_add_incompatible_raises(self):
        """Adding equations with different LHS raises ValueError"""
        eq1 = Equation(d_dt(n_("H")), sp.Symbol("a"))
        eq2 = Equation(d_dt(n_("He")), sp.Symbol("b"))
        with pytest.raises(ValueError, match="incompatible"):
            eq1 + eq2

    def test_radd_zero(self):
        """0 + eq returns the equation (needed for sum())"""
        eq = Equation(d_dt(n_("H")), sp.Symbol("a"))
        result = 0 + eq
        assert isinstance(result, Equation)
        assert result.rhs == sp.Symbol("a")

    def test_radd_boolean(self):
        """Adding a BooleanAtom (e.g. True/False from sympy) adds 0"""
        eq = Equation(d_dt(n_("H")), sp.Symbol("a"))
        result = eq + sp.true
        assert result.rhs == sp.Symbol("a")

    def test_iadd(self):
        """In-place addition works"""
        lhs = d_dt(n_("H"))
        eq = Equation(lhs, sp.Symbol("a"))
        eq += Equation(lhs, sp.Symbol("b"))
        assert eq.rhs == sp.Symbol("a") + sp.Symbol("b")

    def test_isub(self):
        """In-place subtraction works"""
        lhs = d_dt(n_("H"))
        eq = Equation(lhs, sp.Symbol("a"))
        eq -= Equation(lhs, sp.Symbol("b"))
        assert eq.rhs == sp.Symbol("a") - sp.Symbol("b")

    def test_sum_list(self):
        """sum() over a list of equations works via __radd__(0)"""
        lhs = d_dt(n_("H"))
        eqs = [Equation(lhs, sp.Integer(i)) for i in range(1, 4)]
        result = sum(eqs)
        assert isinstance(result, Equation)
        assert result.rhs == 6  # 1+2+3

    def test_preserves_equation_type(self):
        """Result of operations is still an Equation, not a plain Equality"""
        lhs = d_dt(n_("H"))
        eq = Equation(lhs, sp.Symbol("a")) + Equation(lhs, sp.Symbol("b"))
        assert type(eq) is Equation


# ============================================================
# EquationSystem tests
# ============================================================


class TestEquationSystem:
    def test_auto_init_missing_key(self):
        """Accessing a missing key creates a zero-RHS differential equation"""
        es = EquationSystem()
        eq = es["H"]
        assert isinstance(eq, Equation)
        assert eq.rhs == 0
        assert "H" in es

    def test_add_systems(self):
        """Adding two systems unions keys and sums overlapping values"""
        es1 = EquationSystem()
        es1["H"] = Equation(d_dt(n_("H")), sp.Symbol("a"))

        es2 = EquationSystem()
        es2["H"] = Equation(d_dt(n_("H")), sp.Symbol("b"))
        es2["He"] = Equation(d_dt(n_("He")), sp.Symbol("c"))

        combined = es1 + es2
        assert "H" in combined
        assert "He" in combined
        assert combined["H"].rhs == sp.Symbol("a") + sp.Symbol("b")
        assert combined["He"].rhs == sp.Symbol("c")

    def test_symbols_property(self):
        """symbols property returns all free symbols in the system"""
        es = EquationSystem()
        a, b = sp.symbols("a b")
        es["H"] = Equation(d_dt(n_("H")), a * n_("H"))
        es["He"] = Equation(d_dt(n_("He")), b * n_("He"))
        syms = es.symbols
        assert a in syms
        assert b in syms

    def test_rhs_property(self):
        """rhs returns dict of rhs-lhs for each equation"""
        es = EquationSystem()
        lhs = d_dt(n_("H"))
        es["H"] = Equation(lhs, sp.Symbol("a"))
        rhs_dict = es.rhs
        assert "H" in rhs_dict
        # rhs - lhs
        assert rhs_dict["H"] == sp.Symbol("a") - lhs

    def test_copy_independent(self):
        """copy() produces an independent system"""
        es = EquationSystem()
        es["H"] = Equation(d_dt(n_("H")), sp.Symbol("a"))
        es2 = es.copy()

        # modify original
        es["H"] = Equation(d_dt(n_("H")), sp.Symbol("z"))
        # copy should be unchanged
        assert es2["H"].rhs == sp.Symbol("a")

    def test_subs(self):
        """subs() substitutes throughout all equations"""
        es = EquationSystem()
        a, b = sp.symbols("a b")
        es["H"] = Equation(d_dt(n_("H")), a + b)
        es.subs(a, 2 * b)
        assert es["H"].rhs == 3 * b

    def test_chemical_species(self):
        """chemical_species detects species from n_ and x_ symbols"""
        es = EquationSystem()
        es["H"] = Equation(d_dt(n_("H")), n_("H") * sp.Symbol("k"))
        es["He"] = Equation(d_dt(n_("He")), x_("He") * sp.Symbol("q"))
        species = es.chemical_species
        assert "H" in species
        assert "He" in species

    def test_jacobian(self):
        """jacobian returns correct symbolic derivatives"""
        es = EquationSystem()
        a = sp.Symbol("a")
        es["H"] = Equation(d_dt(n_("H")), a * n_("H") ** 2)
        jac = es.jacobian
        # d/d(n_H) of a*n_H^2 = 2*a*n_H
        assert jac["H"][n_("H")] == 2 * a * n_("H")

    def test_conservation_reduction_charge_neutrality(self):
        """do_conservation_reductions eliminates e- via charge neutrality"""
        es = EquationSystem()
        k_ion = sp.Symbol("k_ion")
        k_rec = sp.Symbol("k_rec")
        # Ionization/recombination: H <-> H+ + e-
        # n_e- must appear in RHS for the reduction to detect it
        es["H"] = Equation(d_dt(n_("H")), -k_ion * n_("H") + k_rec * n_("H+") * n_("e-"))
        es["H+"] = Equation(d_dt(n_("H+")), k_ion * n_("H") - k_rec * n_("H+") * n_("e-"))
        es["e-"] = Equation(d_dt(n_("e-")), k_ion * n_("H") - k_rec * n_("H+") * n_("e-"))
        es["heat"] = Equation(d_dt(n_("heat")), 0)

        es.do_conservation_reductions(time_dependent_vars=[])

        # e- should have been eliminated via charge neutrality
        assert "e-" not in es
        # x_e- should have been substituted with x_H+ (charge neutrality for singly-ionized)
        assert "H+" in es

    def test_conservation_reduction_atom_conservation(self):
        """do_conservation_reductions eliminates atom species via conservation"""
        es = EquationSystem()
        k = sp.Symbol("k")
        # H -> H+ + e- : atom H is conserved between H and H+
        es["H"] = Equation(d_dt(n_("H")), -k * n_("H"))
        es["H+"] = Equation(d_dt(n_("H+")), k * n_("H"))
        es["e-"] = Equation(d_dt(n_("e-")), k * n_("H"))
        es["heat"] = Equation(d_dt(n_("heat")), 0)

        es.do_conservation_reductions(time_dependent_vars=[])

        # H (the neutral atom) should be eliminated — solved via conservation of total H
        assert "H" not in es


# ============================================================
# Process composition tests
# ============================================================


class TestProcessComposition:
    def test_add_two_processes(self):
        """Adding two processes merges their networks"""
        p1 = Process(name="proc1")
        p1.heat = sp.Symbol("Q1")

        p2 = Process(name="proc2")
        p2.heat = sp.Symbol("Q2")

        combined = p1 + p2
        assert combined.network["heat"].rhs == sp.Symbol("Q1") + sp.Symbol("Q2")

    def test_add_preserves_subprocesses(self):
        """Adding processes concatenates subprocess lists"""
        p1 = Process(name="proc1")
        p2 = Process(name="proc2")
        combined = p1 + p2
        assert len(combined.subprocesses) == 2

    def test_add_merges_bibliography(self):
        """Adding processes merges bibliographies"""
        p1 = Process(name="proc1", bibliography=["Paper A"])
        p2 = Process(name="proc2", bibliography=["Paper B"])
        combined = p1 + p2
        assert "Paper A" in combined.bibliography
        assert "Paper B" in combined.bibliography

    def test_add_zero(self):
        """0 + process returns the process (needed for sum())"""
        p = Process(name="test")
        p.heat = sp.Symbol("Q")
        result = 0 + p
        assert result.network["heat"].rhs == sp.Symbol("Q")

    def test_sum_processes(self):
        """sum() over a list of processes works"""
        processes = [Process(name=f"p{i}") for i in range(3)]
        for i, p in enumerate(processes):
            p.heat = sp.Symbol(f"Q{i}")
        combined = sum(processes)
        assert combined.network["heat"].rhs == sp.Symbol("Q0") + sp.Symbol("Q1") + sp.Symbol("Q2")

    def test_heat_setter_updates_network(self):
        """Setting heat automatically updates network['heat']"""
        p = Process(name="test")
        p.heat = sp.Symbol("Q")
        assert p.network["heat"].rhs == sp.Symbol("Q")

        p.heat = sp.Symbol("Q2")
        assert p.network["heat"].rhs == sp.Symbol("Q2")

    def test_combined_name(self):
        """Combined process name reflects composition"""
        p1 = Process(name="A")
        p2 = Process(name="B")
        combined = p1 + p2
        assert "A" in combined.name
        assert "B" in combined.name

    def test_compose_thermal_processes(self):
        """Composing ThermalProcess instances combines their heating rates"""
        tp1 = ThermalProcess(sp.Symbol("Gamma_pe"), name="photoelectric")
        tp2 = ThermalProcess(-sp.Symbol("Lambda_CII"), name="CII cooling")
        combined = tp1 + tp2
        assert combined.network["heat"].rhs == sp.Symbol("Gamma_pe") - sp.Symbol("Lambda_CII")

    def test_network_equations_merge(self):
        """Networks with overlapping species equations sum correctly"""
        p1 = Process(name="ionization")
        k1 = sp.Symbol("k1")
        p1.network["H"] = Equation(d_dt(n_("H")), -k1 * n_("H"))

        p2 = Process(name="recombination")
        k2 = sp.Symbol("k2")
        p2.network["H"] = Equation(d_dt(n_("H")), k2 * n_("H+") * n_("e-"))

        combined = p1 + p2
        expected_rhs = -k1 * n_("H") + k2 * n_("H+") * n_("e-")
        assert sp.simplify(combined.network["H"].rhs - expected_rhs) == 0

    def test_network_disjoint_species(self):
        """Networks with disjoint species both appear in combined network"""
        p1 = Process(name="p1")
        p1.network["H"] = Equation(d_dt(n_("H")), sp.Symbol("a"))

        p2 = Process(name="p2")
        p2.network["He"] = Equation(d_dt(n_("He")), sp.Symbol("b"))

        combined = p1 + p2
        assert "H" in combined.network
        assert "He" in combined.network
