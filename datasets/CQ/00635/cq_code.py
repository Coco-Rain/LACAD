import cadquery as cq

result = (
cq.Workplane('XY')
.circle(8.425)
.extrude(3)
.faces('>Z')
.circle(6.125)
.extrude(18.0)
.faces('>Z')
.circle(4.725)
.cutThruAll()
.edges('>Z')
.chamfer(0.5)
)
cq.exporters.export(result, 'GT.stl')