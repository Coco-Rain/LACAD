import cadquery as cq

result = (
cq.Workplane("XY")
.box(50, 40, 25)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 12.5))
.ellipse(30, 20)
.extrude(15)
)
.faces(">Z")
.workplane()
.cboreHole(10, 15, 8)
)
cq.exporters.export(result, 'GT.stl')