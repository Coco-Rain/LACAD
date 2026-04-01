import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 10)
.extrude(5)
.faces(">Z")
.workplane()
.hole(3)
)
cq.exporters.export(result, 'GT.stl')