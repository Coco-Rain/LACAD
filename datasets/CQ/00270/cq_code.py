import cadquery as cq

cap = (
cq.Workplane('XY')
.circle(38)
.extrude(38)
.edges('>Z').fillet(3.0)
.edges('<Z').chamfer(3.0)
)
hole = cq.Workplane('XY').circle(32).extrude(28)
result = cap.cut(hole)
cq.exporters.export(result, 'GT.stl')