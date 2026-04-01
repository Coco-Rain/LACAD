import cadquery as cq

result = (
cq.Workplane("XY")
.circle(8)
.extrude(12)
.faces(">Z")
.workplane()
.polygon(5, 6)
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')