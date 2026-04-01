import cadquery as cq

result = (
cq.Workplane("XY")
.circle(12)
.extrude(8)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 8))
.polygon(5, 10)
.extrude(6)
)
.faces(">Z")
.workplane()
.cboreHole(5, 8, 4, 10)
)
cq.exporters.export(result, 'GT.stl')