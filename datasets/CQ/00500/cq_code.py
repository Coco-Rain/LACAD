import cadquery as cq

result = (
cq.Workplane('XY')
.box(20, 9, 2)
.edges('|Z').fillet(4)
.edges('|X').fillet(0.5)
.faces('>Z').workplane()
.center(5.5, 0.0)
.circle(4.5)
.circle(2.3)
.extrude(7.0)
.edges('>Z').fillet(0.2)
)
cq.exporters.export(result, 'GT.stl')