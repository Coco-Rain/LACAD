import cadquery as cq

result = (
cq.Workplane("XY")
.rect(15, 6.8)
.extrude(3.0)
.workplane(1.5, origin = (0, -2.0))
.rect(15, 3.8)
.extrude(18)
.edges("|Y")
.edges(">Z")
.fillet(4)
)
cq.exporters.export(result, 'GT.stl')