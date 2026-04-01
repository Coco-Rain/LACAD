import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 6)
.faces(">Z")
.workplane()
.polygon(5, 8)
.extrude(7)
.edges("|Z")
.fillet(1.5)
)
cq.exporters.export(result, 'GT.stl')