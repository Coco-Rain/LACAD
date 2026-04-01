import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 10, 2)
.faces(">Z")
.workplane()
.moveTo(-6, 0)
.radiusArc((6, 0), 8)
.lineTo(6, -3)
.radiusArc((-6, -3), 8)
.close()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')