import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 37, 70)
.edges("|Z")
.fillet(7)
.faces("<Z")
.circle(9)
.extrude(-10)
)
cq.exporters.export(result, 'GT.stl')