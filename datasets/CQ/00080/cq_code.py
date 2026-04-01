import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 8)
.union(
cq.Workplane("XY")
.transformed(offset=(10, -5, 4))
.cylinder(12, 6)
)
.faces(">Z").workplane()
.slot2D(18, 4)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')