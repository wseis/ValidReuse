import unittest
import numpy as np
from streamlit.testing.v1 import AppTest


class VolumeUITests(unittest.TestCase):
    def test_examples_edits_clear_and_legacy(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        self.assertFalse(at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,3)
        key='vol_editor_ecoli_out_1'
        at.session_state[key]={'edited_rows':{'0':{'value':4}},'added_rows':[],'deleted_rows':[]}
        at.run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,4)
        at.session_state[key]={'edited_rows':{'0':{'value':5}},'added_rows':[],'deleted_rows':[]}
        at.run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,5)
        at.selectbox[0].set_value('Sporen C. perfringens').run()
        at.selectbox[0].set_value('E. coli').run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,5)
        at.button[1].click().run()
        self.assertTrue(at.session_state['vol_ecoli_out'].empty)
        at.sidebar.radio[0].set_value('Bisheriger Methodenvergleich').run()
        self.assertFalse(at.exception)

    def test_missing_assumption_blocks_fitting(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        at.button[0].click().run()
        at.button[2].click().run()
        self.assertTrue(at.error)
        self.assertNotIn('vol_result_ecoli',at.session_state)

    def test_documentation_workspace_renders_every_section(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        at.sidebar.radio[0].set_value('Dokumentation und Formeln').run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        # The volume calculator must reproduce the documented worked example.
        formulas='\n'.join(block.value for block in at.latex)
        self.assertIn(r'V_{\mathrm{eff}}',formulas)
        self.assertIn(r'\log_{10}',formulas)
        metrics={m.label:m.value for m in at.metric}
        self.assertEqual(metrics[r'$V_{\mathrm{eff}}$ (mL)'],'1875')
        self.assertEqual(metrics[r'$c_{\mathrm{Ablauf}}$ (je 100 mL)'],'0.16')
        # The guide's own direction of travel must be visible, not just the ratio.
        self.assertEqual(metrics[r'Bezugsfaktor $E$'],'18.75')
        for equation in [r'\text{(4-1)}',r'\text{(4-2)}',r'\text{(4-3)}',r'\text{(5-1)}',r'\text{(5-2)}']:
            self.assertIn(equation,formulas)
        self.assertIn('18{,}75',formulas)

    def test_count_workspace_shows_reference_factor_and_raised_influent(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual({m.label:m.value for m in at.metric}['Bezugsfaktor'],'18.75')
        columns=set().union(*[set(frame.value.columns) for frame in at.dataframe])
        self.assertIn('Bezugsfaktor V_eff/100 mL',columns)
        self.assertIn('Zulauf bezogen auf V_eff, Gl. 4-3',columns)
        self.assertIn('Count je 100 mL (nur Bericht)',columns)
        self.assertIn('nicht eine je 100 mL','\n'.join(block.value for block in at.info))

    def test_conditional_candidates_are_selectable_and_swap_the_assumption_widget(self):
        at=AppTest.from_file('app.py',default_timeout=60).run()
        at.button[0].click().run()
        # Unpaired stays the default so historical influent values remain usable.
        self.assertEqual(at.multiselect[0].value,
                         ['Poisson-Lognormal','Negativbinomial (Poisson-Gamma)'])
        at.multiselect[0].set_value(['Bedingt · Poisson-Lognormal']).run()
        self.assertFalse(at.exception)
        self.assertEqual([r.label for r in at.radio if 'Kopplung' in r.label],
                         ['Kopplung b der bedingten Kandidaten'])
        # The independence confirmation applies to the unpaired candidates only.
        self.assertEqual([c for c in at.checkbox if 'bestätige' in c.label],[])
        at.multiselect[0].set_value(['Poisson-Lognormal']).run()
        self.assertEqual([r for r in at.radio if 'Kopplung' in r.label],[])
        self.assertEqual(len([c for c in at.checkbox if 'bestätige' in c.label]),1)

    def test_documentation_shows_the_conditional_fit_flow(self):
        at=AppTest.from_file('app.py',default_timeout=60).run()
        at.sidebar.radio[0].set_value('Dokumentation und Formeln').run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get('graphviz_chart')),1)
        pseudocode=at.get('code')[0].value
        for step in ['Normal(mu_in[s], sd_in[s])','q10[s]','L_alpha']:
            self.assertIn(step,pseudocode)
        formulas='\n'.join(block.value for block in at.latex)
        self.assertIn(r'(1-b)^{2}',formulas)


    def test_recovery_sensitivity_shifts_the_bound_by_log10_of_the_recovery_ratio(self):
        at=AppTest.from_file('app.py',default_timeout=900).run()
        at.button[0].click().run()
        for field in at.number_input:
            if field.label.startswith('Ziehungen'): field.set_value(300)
            if field.label.startswith('Warmup'): field.set_value(300)
        at.run()
        at.multiselect[0].set_value(['Poisson-Lognormal']).run()
        [c for c in at.checkbox if 'bestätige' in c.label][0].set_value(True).run()
        [b for b in at.button if 'berechnen' in b.label][0].click().run()
        self.assertFalse(at.exception)
        bound=at.session_state['vol_result_ecoli']['result']['results']['Poisson-Lognormal']['L_alpha']
        self.assertEqual(len(at.get('vega_lite_chart')),1)
        table=[f.value for f in at.dataframe if 'Wiederfindung' in list(f.value.columns)][0]
        shown=dict(zip(table['Wiederfindung'],table['Poisson-Lognormal']))
        # The example data assume 50 % recovery; the curve is a pure log10 shift from there.
        self.assertAlmostEqual(shown['50 %'],bound,places=9)
        self.assertAlmostEqual(shown['5 %'],bound-1.,places=9)
        self.assertAlmostEqual(shown['100 %'],bound+np.log10(2.),places=9)
        critical=[m.value for m in at.metric if m.label.startswith('Grenzwiederfindung')][0]
        self.assertTrue(critical.endswith('%'),critical)



if __name__=='__main__': unittest.main()
