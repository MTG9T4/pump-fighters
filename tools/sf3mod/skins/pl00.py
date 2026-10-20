"""pl00 Gill -> PILL (final boss).

Gill's bank carries a few drawings of his body that the core takes for effects (they are loose pieces of a
multi-sprite pose, or drawn in his fire / ice colours): the blue half-body (518-519), the resurrection pieces (head,
hands and feet assembling, 730-735) and the robed intro figure reading the book (960-963). The rule is no original
character on screen, so they become the mascot: a fitted / borrowed mascot body, and a loose mascot arm for the two
little hand pieces (614-615). The black winged silhouettes (1182-1210) and the fire / ice effects keep their drawing.
"""
PIECES = {'body': [(518, 519), (730, 735), (960, 963)], 'arm': [(614, 615)]}

# 730 (the first resurrection frame) is too broken up for the completion pass: it borrows the pose the next frames
# get (729, standing) and, like them, shows the mascot only through the drawing's pieces
BORROW = {730: 729}
# Effects the completion pass drew as PILL (fire / ice balls and bursts, fire streaks, the flame pieces after the
# Seraphic Wing): they keep their drawing
EFFECT = [(768, 768), (771, 771), (800, 860), (921, 921), (931, 931), (1075, 1078), (1251, 1282)]


def fix_pose(p, J, rec, ctx):
    if p == 730:
        J['_treat'] = {'mask_src': True}; J['_src'] = (ctx['bank_name'], p)
    return J
