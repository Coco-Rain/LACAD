import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 8)
.union(
cq.Workplane("XZ")
.transformed(offset=(0, 4, 6))
.polygon(6, 5)
.extrude(3)
)
.faces(">Y")
.workplane()
.cboreHole(2.5, 5, 2)
)
cq.exporters.export(result, 'GT.stl')