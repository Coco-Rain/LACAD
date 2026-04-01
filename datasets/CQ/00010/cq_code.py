import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 30, 12)
.faces(">Z")
.workplane()
.cylinder(18, 8)
.faces("<Z")
.workplane()
.transformed(offset=(15, 10, 0))
.cskHole(5.5, 8, 9, 4)
)
cq.exporters.export(result, 'GT.stl')