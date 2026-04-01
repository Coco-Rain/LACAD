import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 10)
.faces(">Z").workplane()
.polygon(nSides=6, diameter=15)
.extrude(5)
.faces(">Z").workplane()
.center(10, 0)
.radiusArc(endPoint=(10, 5), radius=8)
.lineTo(10, 0)
.close()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')