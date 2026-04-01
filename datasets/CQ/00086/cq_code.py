import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 8)
.faces(">Z")
.workplane()
.polygon(5, 6)
.cutBlind(-5)
)
cq.exporters.export(result, 'GT.stl')