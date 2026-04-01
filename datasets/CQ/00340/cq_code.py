import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 25)
.extrude(8)
.faces(">Z")
.workplane()
.cylinder(12, 10)
.faces(">Z[-2]")
.workplane()
.hole(6, 16)
)
result
cq.exporters.export(result, 'GT.stl')