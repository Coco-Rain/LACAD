import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
.faces(">Z")
.workplane()
.rect(2, 2, forConstruction=True)
.vertices()
.circle(0.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')