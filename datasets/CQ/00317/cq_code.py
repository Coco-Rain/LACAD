import cadquery as cq

result = (
cq.Workplane('XY', origin=(43, -53, 24))
.circle(25)
.extrude(2)
.tag('base')
.faces('>Z')
.workplane()
.hole(46, 2)
.faces('>Z', tag='base')
.workplane()
.hole(42)
)
cq.exporters.export(result, 'GT.stl')