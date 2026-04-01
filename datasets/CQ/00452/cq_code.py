import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 30, 10)
.faces(">Z")
.workplane()
.transformed(offset=(0, -5, 0))
.moveTo(-8, 0)
.lineTo(8, 0)
.threePointArc((12, 6), (8, 12))
.lineTo(-8, 12)
.threePointArc((-12, 6), (-8, 0))
.close()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')