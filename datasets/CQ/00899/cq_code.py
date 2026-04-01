import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 6)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 6))
.sphere(5)
)
.faces(">Z")
.workplane()
.cboreHole(3.0, 6.0, 2.0, 8.0)
)
cq.exporters.export(result, 'GT.stl')